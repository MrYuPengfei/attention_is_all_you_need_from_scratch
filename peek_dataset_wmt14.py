"""查看 WMT14 数据集信息，并用 PyTorch 原生方式演示数据加载流程。

如果你之前不太熟悉 PyTorch，可以先记住这条数据链路：

    Parquet 文件
      -> Hugging Face datasets 读取为可索引的数据表
      -> torch.utils.data.Dataset 包装成“给我下标，我返回一条样本”的对象
      -> Sampler 决定“按什么顺序取下标”
      -> BatchSampler 决定“把哪些下标凑成一个 batch”
      -> DataLoader 负责多进程、打乱、组 batch、padding
      -> 模型训练循环拿到 src / tgt 两个 tensor

Dataset 与 DataLoader 的分工：
- Dataset 只关心“第 i 条样本是什么”，通常实现 __len__ 和 __getitem__。
- DataLoader 不关心样本怎么读，它负责批量、乱序、多进程、pin_memory 等工程细节。
- collate_fn 负责把一个 batch 的样本合并成模型能吃的 tensor；本脚本里会做 padding。

脚本会输出：
1. Parquet 文件、行数、schema、大小等元信息
2. 原始文本样例与词数统计
3. SentencePiece/BPE 词表信息（如果找到）
4. 编码后的 token 长度统计
5. Dataset / DataLoader / Sampler / random_split / ConcatDataset 等组件的实际运行结果
6. PyTorch 数据加载概念说明与数据集使用方式

用法：
    python peek_dataset_wmt14.py
    python peek_dataset_wmt14.py --lang fr-en --sample-size 5000
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    BatchSampler,
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
    WeightedRandomSampler,
    get_worker_info,
    random_split,
)

from variations_train_val_test import BOS, EOS, PAD

# ======================
# torch.utils.data 常用概念速览（先读注释，再看下面代码）
# ======================
# Dataset（map-style）：像 Python list 一样用 dataset[i] 取第 i 条样本。
# IterableDataset（stream-style）：用 for item in dataset 顺序读取，适合流式的大数据。
#
# Sampler：只产生“下标顺序”，不返回样本本身。
#   - SequentialSampler: 0,1,2,...
#   - RandomSampler: 随机下标；replacement=True 表示同一个下标可重复抽到。
#   - SubsetRandomSampler: 只在给定下标集合里随机抽。
#   - WeightedRandomSampler: 按权重抽下标，常用于类别/样本不平衡。
#   - BatchSampler: 把若干下标打包成一个 batch 的下标列表。
#
# DataLoader：
#   - shuffle=True 时，内部自动使用 RandomSampler。
#   - batch_size 决定每个 batch 多少条样本；drop_last 丢弃最后不足一个 batch 的部分。
#   - num_workers>0 会用多子进程并行读数据；IterableDataset 需要手动按 worker 分片。
#   - pin_memory=True 会把数据放到锁页内存，通常配合 GPU 使用，能加快 CPU->GPU 拷贝。
#   - collate_fn 决定如何把一个 batch 的样本拼成最终返回给模型的对象。
#
# 本脚本中的特殊点：
# - WMT14Dataset 返回 token id 张量，长度不一致，所以需要 pad_sequence 补齐。
# - WMT14IterableDataset 用 get_worker_info() 给不同 worker 分配不同区间，避免重复读同样数据。

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
LANG_PAIRS = {
    "de-en": ("de", "en"),
    "fr-en": ("fr", "en"),
}
SPECIAL_TOKENS = {"PAD": PAD, "BOS": BOS, "EOS": EOS}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANG_PAIRS, default="de-en")
    parser.add_argument("--data-dir", type=Path, default=None, help="默认 dataset/wmt14/<lang>")
    parser.add_argument("--sample-size", type=int, default=5000, help="统计时采样多少条句对")
    parser.add_argument("--show-samples", type=int, default=2, help="打印多少条原始样例")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--spm-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=9527)
    return parser.parse_args()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def kv(name, value):
    print(f"{name:<34}: {value}")


def fmt_size(num_bytes):
    return f"{num_bytes / 1024**3:.3f} GB ({num_bytes:,} bytes)"


def sample_indices(total, limit):
    """在 [0, total) 中均匀采样，避免遍历全量数据。"""
    if total <= limit:
        return list(range(total))
    step = total / limit
    return sorted({min(total - 1, int(i * step)) for i in range(limit)})


def describe_values(values):
    """输出常见统计指标。"""
    values = sorted(values)
    if not values:
        return {}
    n = len(values)

    def percentile(q):
        pos = (n - 1) * q
        low, high = int(pos), min(int(pos) + 1, n - 1)
        return values[low] + (values[high] - values[low]) * (pos - low)

    return {
        "count": n,
        "min": values[0],
        "max": values[-1],
        "mean": round(sum(values) / n, 3),
        "p50": round(percentile(0.50), 3),
        "p90": round(percentile(0.90), 3),
        "p95": round(percentile(0.95), 3),
        "p99": round(percentile(0.99), 3),
    }


class WMT14Dataset(Dataset):
    """Map-style Dataset：按 index 返回一条句对。

    继承 torch.utils.data.Dataset 时，通常至少实现两个方法：
    - __len__: 告诉 DataLoader 一共有多少条样本。
    - __getitem__: DataLoader 给出下标 index，你返回对应样本。

    若传入 SentencePiece 模型，则返回 token id 张量；否则返回原始字符串。
    """

    def __init__(self, hf_dataset, src_lang, tgt_lang, sp=None, max_len=100):
        self.data = hf_dataset       # Hugging Face datasets.Dataset，支持 dataset[i]
        self.src_lang = src_lang     # 源语言，例如 de
        self.tgt_lang = tgt_lang     # 目标语言，例如 en
        self.sp = sp                 # SentencePiece 处理器；None 表示不做分词
        self.max_len = max_len       # 包含 BOS/EOS 的最大 token 数

    def __len__(self):
        # DataLoader 会用这个值决定 Sampler 能产生哪些下标。
        return len(self.data)

    def __getitem__(self, index):
        # Hugging Face 数据每行是 {"translation": {"de": "...", "en": "..."}}。
        pair = self.data[index]["translation"]
        src_text, tgt_text = pair[self.src_lang], pair[self.tgt_lang]

        # 没传 sp 时返回原始字符串，便于直接查看文本内容。
        if self.sp is None:
            return src_text, tgt_text

        # 有 sp 时做 BPE 编码：
        # BOS + 最多 max_len-2 个真实 token + EOS，总长度不超过 max_len。
        # transformer 模型需要 LongTensor 形式的 token id。
        src_ids = [BOS] + self.sp.encode(src_text, out_type=int)[: self.max_len - 2] + [EOS]
        tgt_ids = [BOS] + self.sp.encode(tgt_text, out_type=int)[: self.max_len - 2] + [EOS]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class WMT14IterableDataset(IterableDataset):
    """Iterable-style Dataset：演示 get_worker_info 的多 worker 分片。

    与 map-style Dataset 不同，IterableDataset 没有 __getitem__，
    而是通过 __iter__ 顺序 yield 样本。DataLoader 会不断调用 next() 取值。
    """

    def __init__(self, hf_dataset, src_lang, tgt_lang, sp=None, max_len=100):
        self.data = hf_dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sp = sp
        self.max_len = max_len

    def __iter__(self):
        # 如果 DataLoader 使用 num_workers > 0，会有多个子进程同时执行 __iter__。
        # 如果不分片，每个 worker 都会从头读到尾，导致同一条样本被重复读取。
        worker_info = get_worker_info()
        total = len(self.data)
        if worker_info is None:
            # num_workers=0：只有主进程，自己读全部数据。
            start, end = 0, total
        else:
            # 多 worker：按 worker id 把数据区间切成互不重叠的几段。
            per_worker = (total + worker_info.num_workers - 1) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)

        for index in range(start, end):
            pair = self.data[index]["translation"]
            if self.sp is None:
                yield pair[self.src_lang], pair[self.tgt_lang]
            else:
                src_ids = [BOS] + self.sp.encode(pair[self.src_lang], out_type=int)[: self.max_len - 2] + [EOS]
                tgt_ids = [BOS] + self.sp.encode(pair[self.tgt_lang], out_type=int)[: self.max_len - 2] + [EOS]
                yield torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_tokens(batch):
    """把变长 token 序列 pad 成 batch tensor。

    DataLoader 每次会把样本组成 list，例如：
        [(src1, tgt1), (src2, tgt2), ...]
    zip(*batch) 把它拆成：
        srcs = (src1, src2, ...)
        tgts = (tgt1, tgt2, ...)
    每条句子长度不同，不能直接 stack 成矩阵，所以要用 pad_sequence 补 PAD。
    batch_first=True 让形状变成 [batch_size, max_len]，这是 Transformer 常用形状。
    """
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD),
        pad_sequence(tgts, batch_first=True, padding_value=PAD),
    )


def collate_raw(batch):
    """原始字符串 batch：直接分别收集 src / tgt，不需要 padding。"""
    srcs, tgts = zip(*batch)
    return list(srcs), list(tgts)


def find_spm_model(args, data_dir):
    """自动查找可用的 SentencePiece 模型。"""
    if args.spm_model:
        return args.spm_model if args.spm_model.exists() else None

    patterns = [
        str(ROOT / "results" / "*" / "spm" / f"{args.lang}.bpe.*.model"),
        str(data_dir / "spm" / "*.model"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(Path(path) for path in glob.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def report_parquet_files(data_dir):
    """输出每个 split 的 Parquet 文件元信息。"""
    import pyarrow.parquet as pq

    summary = {}
    for split in SPLITS:
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        rows, row_groups, columns, sizes = 0, 0, 0, 0
        details = []
        for file in files:
            meta = pq.ParquetFile(file).metadata
            rows += meta.num_rows
            row_groups += meta.num_row_groups
            columns = max(columns, meta.num_columns)
            sizes += file.stat().st_size
            details.append(f"{file.name}: rows={meta.num_rows:,} groups={meta.num_row_groups}")
        summary[split] = {"files": files, "rows": rows, "bytes": sizes}

        kv(f"{split} 文件数", len(files))
        kv(f"{split} 句对数", f"{rows:,}")
        kv(f"{split} 文件大小", fmt_size(sizes))
        kv(f"{split} row groups", row_groups)
        kv(f"{split} columns", columns)
        for line in details:
            print(f"    - {line}")
    return summary


def report_raw_samples(hf_data, args, src_lang, tgt_lang):
    """打印原始文本样例。"""
    section("原始文本样例")
    for split in SPLITS:
        dataset = hf_data[split]
        for i in range(min(args.show_samples, len(dataset))):
            pair = dataset[i]["translation"]
            print(f"[{split} #{i}]")
            print(f"  {src_lang}: {pair[src_lang][:300]}")
            print(f"  {tgt_lang}: {pair[tgt_lang][:300]}")


def report_raw_stats(hf_data, args, src_lang, tgt_lang):
    """统计原始文本的词数分布。"""
    section("原始文本词数统计（按空白切分）")
    for split in SPLITS:
        dataset = hf_data[split]
        indices = sample_indices(len(dataset), args.sample_size)
        src_words, tgt_words = [], []
        for i in indices:
            pair = dataset[i]["translation"]
            src_words.append(len(pair[src_lang].split()))
            tgt_words.append(len(pair[tgt_lang].split()))
        kv(f"{split} {src_lang} 词数", json.dumps(describe_values(src_words), ensure_ascii=False))
        kv(f"{split} {tgt_lang} 词数", json.dumps(describe_values(tgt_words), ensure_ascii=False))


def report_spm(spm_path, sp, hf_data, args, src_lang, tgt_lang):
    """输出 BPE 词表信息及编码后的 token 长度统计。"""
    section("SentencePiece / BPE 信息")
    if sp is None or spm_path is None:
        print("未找到 SentencePiece 模型，跳过 token 统计。")
        print("可用 --spm-model 指定，例如 results/base/spm/de-en.bpe.32768.model")
        return

    kv("spm 模型路径", spm_path)
    kv("spm 文件大小", fmt_size(spm_path.stat().st_size))
    kv("vocab size", sp.get_piece_size())
    kv("特殊 token", SPECIAL_TOKENS)

    for split in SPLITS:
        dataset = hf_data[split]
        indices = sample_indices(len(dataset), args.sample_size)
        src_tokens, tgt_tokens = [], []
        for i in indices:
            pair = dataset[i]["translation"]
            src_tokens.append(len(sp.encode(pair[src_lang], out_type=int)) + 2)
            tgt_tokens.append(len(sp.encode(pair[tgt_lang], out_type=int)) + 2)
        kv(f"{split} {src_lang} token 长度（含 BOS/EOS）",
           json.dumps(describe_values(src_tokens), ensure_ascii=False))
        kv(f"{split} {tgt_lang} token 长度（含 BOS/EOS）",
           json.dumps(describe_values(tgt_tokens), ensure_ascii=False))


def print_concepts():
    """打印 PyTorch 数据加载概念说明，帮助新手阅读后面的输出。"""
    section("PyTorch 数据加载概念速览")
    print("""
Dataset            : 定义“第 i 条样本是什么”。实现 __len__ 和 __getitem__。
IterableDataset    : 用 for item in dataset 顺序流式读取，适合超大/流式数据。
Sampler            : 只产生访问下标，不返回样本。决定“读取顺序”。
BatchSampler       : 把多个下标组成一个 batch 的下标列表。
DataLoader         : 把 Dataset + Sampler + collate_fn 组合起来，负责批量/乱序/多进程。
collate_fn         : 把一个 batch 的样本合并成最终 tensor；这里用 pad_sequence 补齐变长句子。
num_workers        : 用几个子进程读数据；>0 时 IterableDataset 必须自己分片。
pin_memory         : 使用锁页内存，GPU 训练时通常设为 True，可加快 Host->GPU 拷贝。
drop_last          : 最后不足一个 batch 的样本是否丢弃。
shuffle=True       : DataLoader 内部使用 RandomSampler；与显式 sampler 不能同时指定。
""")


def describe_loader(name, loader):
    """输出 DataLoader 的配置参数。"""
    kv(f"{name}.batch_size", loader.batch_size)
    kv(f"{name}.drop_last", loader.drop_last)
    kv(f"{name}.num_workers", loader.num_workers)
    kv(f"{name}.pin_memory", loader.pin_memory)
    kv(f"{name}.sampler", type(loader.sampler).__name__)
    kv(f"{name}.batch_sampler", type(loader.batch_sampler).__name__)
    kv(f"{name}.dataset_len", len(loader.dataset))


def first_batch(loader):
    """取一个 batch，失败时返回 None。"""
    try:
        return next(iter(loader))
    except Exception as error:
        print(f"取 batch 失败: {error}")
        return None


def batch_shape_text(batch):
    if batch is None:
        return "无"
    return ", ".join(
        f"tensor{tuple(item.shape)}" if torch.is_tensor(item) else f"list[{len(item)}]"
        for item in batch
    )


def report_torch_utils(hf_data, args, src_lang, tgt_lang, sp):
    """实际运行 torch.utils.data 常用组件，展示加载与使用方式。

    这里只用 256 条验证集样本做演示，避免为了看 API 效果而加载/遍历全量数据。
    """
    section("torch.utils.data 组件演示")

    # 固定随机种子，让 RandomSampler / random_split 的结果可复现。
    torch.manual_seed(args.seed)
    val = hf_data["validation"].select(range(min(256, len(hf_data["validation"]))))
    train_small = hf_data["train"].select(range(min(256, len(hf_data["train"]))))

    dataset = WMT14Dataset(val, src_lang, tgt_lang, sp, args.max_len)
    kv("Dataset 类型", type(dataset).__name__)
    kv("Dataset 长度", len(dataset))
    kv("dataset[0] 类型", type(dataset[0]).__name__)

    pin = torch.cuda.is_available()
    collate = collate_tokens if sp is not None else collate_raw

    # 1. 最基础的 DataLoader：shuffle=False，内部使用 SequentialSampler。
    # 训练时一般 shuffle=True；验证/测试时通常 shuffle=False，方便结果对齐。
    seq_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=pin,
    )
    describe_loader("Sequential DataLoader", seq_loader)
    kv("Sequential batch", batch_shape_text(first_batch(seq_loader)))

    # 2. shuffle=True 会自动创建 RandomSampler。
    # 注意：DataLoader 里 shuffle=True 和显式 sampler=... 不能同时使用。
    random_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=pin,
    )
    describe_loader("Shuffle DataLoader", random_loader)
    kv("Shuffle batch", batch_shape_text(first_batch(random_loader)))

    # 3. Sampler 本身是可迭代对象，迭代它得到的是“下标”，不是样本。
    # 显式写 Sampler 比 shuffle=True 更灵活，例如配合自定义采样策略。
    seq_sampler = SequentialSampler(dataset)
    sampler_loader = DataLoader(dataset, sampler=seq_sampler, batch_size=args.batch_size, collate_fn=collate)
    kv("SequentialSampler 前 5 个索引", list(seq_sampler)[:5])
    kv("Sampler DataLoader batch", batch_shape_text(first_batch(sampler_loader)))

    # 4. replacement=True 表示有放回采样：同一个样本可能被抽到多次。
    # num_samples=8 表示这一轮采样器一共产出 8 个下标（不一定等于数据集长度）。
    repeat_sampler = RandomSampler(dataset, num_samples=8, replacement=True)
    kv("RandomSampler(replacement=True) 索引", list(repeat_sampler))
    repeat_loader = DataLoader(dataset, sampler=repeat_sampler, batch_size=4, collate_fn=collate)
    kv("RandomSampler batch", batch_shape_text(first_batch(repeat_loader)))

    # 5. Subset 只是“原 Dataset 的一部分下标视图”，不会复制底层数据。
    # 常用于从一个大 Dataset 中截取一小部分做快速验证。
    subset = Subset(dataset, indices=list(range(min(16, len(dataset)))))
    subset_loader = DataLoader(subset, batch_size=4, collate_fn=collate)
    kv("Subset 长度", len(subset))
    kv("Subset batch", batch_shape_text(first_batch(subset_loader)))

    # 6. SubsetRandomSampler
    subset_random = SubsetRandomSampler(list(range(min(16, len(dataset)))))
    kv("SubsetRandomSampler 前 8 个索引", list(subset_random)[:8])
    sub_random_loader = DataLoader(dataset, sampler=subset_random, batch_size=4, collate_fn=collate)
    kv("SubsetRandomSampler batch", batch_shape_text(first_batch(sub_random_loader)))

    # 7. WeightedRandomSampler
    weights = torch.linspace(0.1, 1.0, steps=len(dataset), dtype=torch.double)
    weighted_sampler = WeightedRandomSampler(weights, num_samples=8, replacement=True)
    kv("WeightedRandomSampler 权重范围",
       f"{float(weights.min()):.2f} ~ {float(weights.max()):.2f}")
    weighted_loader = DataLoader(dataset, sampler=weighted_sampler, batch_size=4, collate_fn=collate)
    kv("WeightedRandomSampler batch", batch_shape_text(first_batch(weighted_loader)))

    # 8. BatchSampler
    batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=4, drop_last=False)
    kv("BatchSampler 前 3 个 batch 索引", [batch[:4] for batch in list(batch_sampler)[:3]])
    batch_sampler_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate)
    kv("BatchSampler DataLoader batch", batch_shape_text(first_batch(batch_sampler_loader)))

    # 9. random_split 把 Dataset 随机切成若干互不相交的 Subset。
    # generator 固定随机种子后，每次运行切分结果一致，方便复现实验。
    n = len(dataset)
    part1, part2 = random_split(dataset, [n // 2, n - n // 2], generator=torch.Generator().manual_seed(args.seed))
    kv("random_split 长度", f"{len(part1)} + {len(part2)}")
    kv("random_split[0] 类型", type(part1[0]).__name__)

    # 10. ConcatDataset
    concat = ConcatDataset([part1, part2])
    kv("ConcatDataset 长度", len(concat))
    concat_loader = DataLoader(concat, batch_size=4, collate_fn=collate)
    kv("ConcatDataset batch", batch_shape_text(first_batch(concat_loader)))

    # 11. IterableDataset 用 __iter__ 顺序产生样本；ChainDataset 把多个 IterableDataset 首尾相接。
    # batch_size=None 表示不做额外组 batch，DataLoader 每次直接返回一个样本。
    iter1 = WMT14IterableDataset(val.select(range(8)), src_lang, tgt_lang, sp, args.max_len)
    iter2 = WMT14IterableDataset(
        train_small.select(range(8)), src_lang, tgt_lang, sp, args.max_len
    )
    chain = ChainDataset([iter1, iter2])
    chain_loader = DataLoader(chain, batch_size=None, num_workers=0)
    chain_batch = first_batch(chain_loader)
    kv("ChainDataset 单条样本类型", type(chain_batch).__name__)
    kv("ChainDataset 单条样本形状/内容", batch_shape_text(chain_batch) if isinstance(chain_batch, tuple) else str(chain_batch)[:120])

    # 12. Dataset / DataLoader 关键属性
    kv("dataset[0] 原始内容", str(dataset[0])[:120])
    kv("DataLoader iterable 类型", type(iter(seq_loader)).__name__)


def report_usage(args, src_lang, tgt_lang, data_dir):
    """打印加载与使用方式。"""
    section("数据集加载与使用方式")
    print("1) 用 Hugging Face datasets 读取 Parquet：")
    print(f"""
from datasets import load_dataset
files = sorted(str(p) for p in Path("{data_dir}").glob("train-*.parquet"))
dataset = load_dataset("parquet", data_files={{"train": files}}, split="train")
pair = dataset[0]["translation"]
print(pair["{src_lang}"], pair["{tgt_lang}"])
""")

    print("2) 包装成 PyTorch Dataset/DataLoader：")
    print(f"""
from torch.utils.data import DataLoader
from peek_dataset_wmt14 import WMT14Dataset, collate_tokens

dataset = WMT14Dataset(dataset, "{src_lang}", "{tgt_lang}", sp=sp, max_len={args.max_len})
loader = DataLoader(dataset, batch_size={args.batch_size}, shuffle=True, collate_fn=collate_tokens)
src, tgt = next(iter(loader))
""")

    print("3) 训练时常用 batch / epoch 关系：")
    batches = 140900 if args.lang == "de-en" else 1136000
    kv("每 epoch batch 数（batch_size=32，约）", f"{batches:,}")
    kv("每 epoch optimizer step（accum=8，约）", f"{batches // args.accum_steps:,}")
    kv("1 个 epoch 的 5% batch 数（约）", f"{int(batches * 0.05):,}")


def main():
    args = parse_args()
    src_lang, tgt_lang = LANG_PAIRS[args.lang]
    data_dir = args.data_dir or ROOT / "dataset" / "wmt14" / args.lang

    section("WMT14 数据集概览")
    kv("报告时间", datetime.now().isoformat(timespec="seconds"))
    kv("语言对", args.lang)
    kv("数据目录", data_dir)
    kv("源语言 / 目标语言", f"{src_lang} -> {tgt_lang}")
    kv("CUDA 可用", torch.cuda.is_available())
    kv("PyTorch 版本", torch.__version__)

    report_parquet_files(data_dir)

    from datasets import load_dataset
    data_files = {
        split: sorted(str(path) for path in data_dir.glob(f"{split}-*.parquet"))
        for split in SPLITS
    }
    hf_data = {
        split: load_dataset("parquet", data_files={split: files}, split=split)
        for split, files in data_files.items()
    }

    for split in SPLITS:
        kv(f"HF {split}.features", hf_data[split].features)
        kv(f"HF {split}.num_rows", len(hf_data[split]))

    report_raw_samples(hf_data, args, src_lang, tgt_lang)
    report_raw_stats(hf_data, args, src_lang, tgt_lang)

    spm_path = find_spm_model(args, data_dir)
    sp = None
    if spm_path is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    report_spm(spm_path, sp, hf_data, args, src_lang, tgt_lang)

    print_concepts()
    report_torch_utils(hf_data, args, src_lang, tgt_lang, sp)
    report_usage(args, src_lang, tgt_lang, data_dir)


if __name__ == "__main__":
    main()
