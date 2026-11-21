"""下载《Attention Is All You Need》论文使用的 WMT14 数据集。

论文使用了 WMT14 En-De 和 WMT14 En-Fr 两个平行语料库。
PyTorch 核心只提供 Dataset/DataLoader 抽象，并不内置 WMT14 下载器，
因此这里用 Hugging Face Hub 下载 Parquet 文件，再通过 datasets 库提供
可直接传给 torch.utils.data.DataLoader 的数据集对象。

依赖：
    pip install datasets huggingface_hub

用法：
    python download_dataset.py                    # 下载英德 + 英法
    python download_dataset.py --langs de-en      # 只下载英德
    python download_dataset.py --endpoint https://huggingface.co
"""

import argparse
import os
from pathlib import Path

REPO_ID = "wmt/wmt14"
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "dataset" / "wmt14"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
# huggingface的国内镜像 "https://hf-mirror.com"
SUPPORTED_LANGS = ("de-en", "fr-en")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=SUPPORTED_LANGS,
        default=list(SUPPORTED_LANGS),
        help="要下载的语言对，默认全部下载",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="数据保存目录",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Hugging Face 端点，默认使用 hf-mirror.com",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行下载线程数",
    )
    return parser.parse_args()


def download_dataset(langs, save_dir, endpoint, max_workers):
    """下载指定语言对的全部 Parquet 文件，支持断点续传。"""
    os.environ["HF_ENDPOINT"] = endpoint

    from huggingface_hub import snapshot_download

    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(save_dir),
        allow_patterns=[f"{lang}/*.parquet" for lang in langs],
        max_workers=max_workers,
    )

    summarize(save_dir, langs)


def summarize(save_dir, langs):
    """打印每个语言对的文件数和磁盘占用。"""
    for lang in langs:
        files = sorted((save_dir / lang).glob("*.parquet"))
        size_gb = sum(path.stat().st_size for path in files) / 1024**3
        print(f"{lang}: {len(files)} 个文件，{size_gb:.2f} GB")

    print(f"数据目录: {save_dir}")


def load_torch_dataset(lang, split="train", data_dir=DEFAULT_SAVE_DIR):
    """把本地 Parquet 数据加载成 PyTorch 可用的 Dataset。

    返回对象可直接传给 torch.utils.data.DataLoader：
        dataset = load_torch_dataset("de-en", "train")
        loader = DataLoader(dataset, batch_size=32)

    文本列仍是字符串；如需 token id，请在此后接分词器处理。
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"不支持的语言对: {lang}")

    from datasets import load_dataset

    files = sorted((data_dir / lang).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"没有找到 {lang}/{split} 数据，请先运行 download_dataset.py"
        )

    dataset = load_dataset(
        "parquet",
        data_files=[str(file) for file in files],
        split="train",
    )
    return dataset.with_format("torch")


if __name__ == "__main__":
    args = parse_args()
    download_dataset(
        langs=args.langs,
        save_dir=args.save_dir,
        endpoint=args.endpoint,
        max_workers=args.max_workers,
    )
