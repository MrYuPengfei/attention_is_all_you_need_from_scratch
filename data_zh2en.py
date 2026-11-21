"""数据集加载与预处理模块。

从 dataset/cmn-eng.txt 读取中英平行语料，构建词表并生成 PyTorch DataLoader。
运行 `python data.py` 可查看语料和词表概况。
"""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split

# 特殊 token：0=填充，1=句子开始，2=句子结束。
pad, sos, eos = 0, 1, 2
# 模型输入/输出的最大长度，包含 sos 和 eos。
seq_len = 30
# 特殊 token 占用 0~2，真实词从 3 开始。
TOKEN_OFFSET = eos + 1

# 使用相对于本文件的路径，避免依赖当前工作目录。
DATA_PATH = Path(__file__).resolve().parent / "cmn-eng.txt"


class WordStat:
    """读取平行语料，并按词频筛选句子和词表。"""

    def __init__(self, path):
        self.pairs = []       # [(中文词列表, 英文词列表), ...]
        self.cmn_counts = {}  # 中文词频
        self.eng_counts = {}  # 英文词频

        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cmn_text, eng_text = line.lower().split("\t")
            cmn, eng = cmn_text.split(), eng_text.split()
            self.pairs.append((cmn, eng))
            self._add_counts(self.cmn_counts, cmn)
            self._add_counts(self.eng_counts, eng)

    @staticmethod
    def _add_counts(counts, words, delta=1):
        """批量增减词频。"""
        for word in words:
            counts[word] = counts.get(word, 0) + delta

    def take(self, min_count=1, max_length=seq_len ):
        """筛选语料，返回 (句子对, 中文词表, 英文词表)。

        过滤掉过长或包含低频词的句子；词表按剩余词频降序排列。
        """
        cmn_counts, eng_counts = self.cmn_counts.copy(), self.eng_counts.copy()

        def is_valid(cmn, eng):
            if max(len(cmn), len(eng)) > max_length:
                return False
            if any(self.cmn_counts[word] <= min_count for word in cmn):
                return False
            if any(self.eng_counts[word] <= min_count for word in eng):
                return False
            return True

        pairs = []
        for cmn, eng in self.pairs:
            if is_valid(cmn, eng):
                pairs.append((cmn, eng))
            else:
                # 与原实现保持一致：被剔除的句子从候选词表计数中扣除。
                self._add_counts(cmn_counts, cmn, -1)
                self._add_counts(eng_counts, eng, -1)

        def sorted_words(counts):
            items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            return [word for word, count in items if count]

        return pairs, sorted_words(cmn_counts), sorted_words(eng_counts)


# 导入本模块时构建一次词表；max_length 为 seq_len - 1，给 sos/eos 留出位置。
cmn_eng, cmn_words, eng_words = WordStat(DATA_PATH).take(
    min_count=1, max_length=seq_len - 1
)


class TranslationDataset(TorchDataset):
    """把词语序列转换成 token id 序列。"""

    def __init__(self, pairs, cmn_words, eng_words):
        # 用 dict 查 id，替代原来的 list.index，更直观也更快。
        cmn_ids = {word: i + TOKEN_OFFSET for i, word in enumerate(cmn_words)}
        eng_ids = {word: i + TOKEN_OFFSET for i, word in enumerate(eng_words)}

        self.data = []
        for cmn, eng in pairs:
            x = [cmn_ids[word] for word in cmn]
            y = [sos, *(eng_ids[word] for word in eng), eos]
            self.data.append(
                (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """把一个 batch 内的变长序列右侧补 pad 到相同长度。"""
    inputs, targets = zip(*batch)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=pad),
        pad_sequence(targets, batch_first=True, padding_value=pad),
    )


def get_dataset(batch_size=32):
    """返回 (训练 DataLoader, 测试 DataLoader)。

    测试集大小等于 batch_size，按单样本加载，便于逐句查看翻译结果。
    """
    dataset = TranslationDataset(cmn_eng, cmn_words, eng_words)
    train_data, test_data = random_split(
        dataset, [len(dataset) - batch_size, batch_size]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(test_data)

    return train_loader, test_loader


if __name__ == "__main__":
    torch.manual_seed(9527)

    print("平行语料条数:", len(cmn_eng))
    print("中文词表大小:", len(cmn_words) + TOKEN_OFFSET)
    print("英文词表大小:", len(eng_words) + TOKEN_OFFSET)
    print("序列最大长度:", seq_len)

    print("\nbatch:\n")
    for train_batch, test_batch in zip(*get_dataset()):
        print(*(tensor.shape for tensor in train_batch))
        print(*test_batch, sep="\n")
        break
