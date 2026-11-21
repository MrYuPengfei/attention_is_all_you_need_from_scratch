"""训练/验证/测试单个 Transformer 变体，并与论文 Table 3 对比。

一次只运行一个变体：
    python variations_train_val_test.py --variant base
    python variations_train_val_test.py --variant A1 --epochs 3

每个 epoch 开始时会打印训练/验证/测试集的首个句子；
训练过程中默认每 100 个 optimizer step 写一条日志；
每个 epoch 结束时计算 PPL/BLEU，并在 PPL 变好时立即保存最优模型。

快速冒烟测试示例：
    python variations_train_val_test.py --variant C1 \
        --max-train-samples 200 --spm-train-samples 200 --vocab-size 200 \
        --epochs 2 --train-steps 1000 --batch-size 2 --accum-steps 1 \
        --max-val-samples 20 --max-test-samples 20 --max-len 32

结果保存目录：results/<variant>/
最优模型：results/<variant>/checkpoints/<variant>_best.pt
日志目录：results/<variant>/logs/
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "dataset" / "wmt14" / "de-en"
DEFAULT_RESULTS_DIR = ROOT / "results"

PAD, BOS, EOS, UNK = 0, 1, 2, 3


@dataclass(frozen=True)
class VariantConfig:
    """Table 3 中一个变体的模型配置。"""

    name: str
    paper_label: str
    N: int = 6
    d_model: int = 512
    d_ff: int = 2048
    h: int = 8
    d_k: int = 64
    d_v: int = 64
    p_drop: float = 0.1
    eps_ls: float = 0.1
    train_steps: int = 100_000
    pos_emb: str = "sinusoidal"
    paper_ppl: float = 0.0
    paper_bleu: float = 0.0
    paper_params: float = 0.0


VARIANTS = {
    "base": VariantConfig("base", "base", paper_ppl=4.92, paper_bleu=25.8, paper_params=65),
    "A1": VariantConfig("A1", "(A)", h=1, d_k=512, d_v=512, paper_ppl=5.29, paper_bleu=24.9),
    "A2": VariantConfig("A2", "(A)", h=4, d_k=128, d_v=128, paper_ppl=5.00, paper_bleu=25.5),
    "A3": VariantConfig("A3", "(A)", h=16, d_k=32, d_v=32, paper_ppl=4.91, paper_bleu=25.8),
    "A4": VariantConfig("A4", "(A)", h=32, d_k=16, d_v=16, paper_ppl=5.01, paper_bleu=25.4),
    "B1": VariantConfig("B1", "(B)", d_k=16, paper_ppl=5.16, paper_bleu=25.1, paper_params=58),
    "B2": VariantConfig("B2", "(B)", d_k=32, paper_ppl=5.01, paper_bleu=25.4, paper_params=60),
    "C1": VariantConfig("C1", "(C)", N=2, paper_ppl=6.11, paper_bleu=23.7, paper_params=36),
    "C2": VariantConfig("C2", "(C)", N=4, paper_ppl=5.19, paper_bleu=25.3, paper_params=50),
    "C3": VariantConfig("C3", "(C)", N=8, paper_ppl=4.88, paper_bleu=25.5, paper_params=80),
    "C4": VariantConfig("C4", "(C)", d_model=256, d_k=32, d_v=32, paper_ppl=5.75, paper_bleu=24.5, paper_params=28),
    "C5": VariantConfig("C5", "(C)", d_model=1024, d_k=128, d_v=128, paper_ppl=4.66, paper_bleu=26.0, paper_params=168),
    "C6": VariantConfig("C6", "(C)", d_ff=1024, paper_ppl=5.12, paper_bleu=25.4, paper_params=53),
    "C7": VariantConfig("C7", "(C)", d_ff=4096, paper_ppl=4.75, paper_bleu=26.2, paper_params=90),
    "D1": VariantConfig("D1", "(D)", p_drop=0.0, paper_ppl=5.77, paper_bleu=24.6),
    "D2": VariantConfig("D2", "(D)", p_drop=0.2, paper_ppl=4.95, paper_bleu=25.5),
    "D3": VariantConfig("D3", "(D)", eps_ls=0.0, paper_ppl=4.67, paper_bleu=25.3),
    "D4": VariantConfig("D4", "(D)", eps_ls=0.2, paper_ppl=5.47, paper_bleu=25.7),
    "E": VariantConfig("E", "(E)", pos_emb="learned", paper_ppl=4.92, paper_bleu=25.7),
    "big": VariantConfig(
        "big", "big", N=6, d_model=1024, d_ff=4096, h=16,
        p_drop=0.3, train_steps=300_000,
        paper_ppl=4.33, paper_bleu=26.4, paper_params=213,
    ),
}

VARIANT_ORDER = list(VARIANTS)


# ======================
# Model
# ======================
class PositionalEncoding(nn.Module):
    """正弦位置编码或可学习位置编码。"""

    def __init__(self, d_model, max_len, mode="sinusoidal"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.embedding = nn.Embedding(max_len, d_model)
        elif mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        else:
            raise ValueError(f"未知位置编码: {mode}")

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        if self.mode == "learned":
            return x + self.embedding(positions).unsqueeze(0)
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """支持 d_k、d_v 与 d_model/h 不同的多头注意力。"""

    def __init__(self, d_model, h, d_k, d_v):
        super().__init__()
        self.h, self.d_k, self.d_v = h, d_k, d_v
        self.W_Q = nn.Linear(d_model, h * d_k)
        self.W_K = nn.Linear(d_model, h * d_k)
        self.W_V = nn.Linear(d_model, h * d_v)
        self.W_O = nn.Linear(h * d_v, d_model)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)

        def split(x, dim):
            return (
                x.view(batch, -1, self.h, dim)
                .transpose(1, 2)
            )

        Q, K, V = split(self.W_Q(q), self.d_k), split(self.W_K(k), self.d_k), split(self.W_V(v), self.d_v)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_v)
        return self.W_O(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.d_model, cfg.h, cfg.d_k, cfg.d_v)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.p_drop)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.p_drop)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.h, cfg.d_k, cfg.d_v)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.h, cfg.d_k, cfg.d_v)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.p_drop)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.p_drop)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.norm3(x + self.dropout(self.ff(x)))


class Encoder(nn.Module):
    def __init__(self, vocab_size, cfg, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(cfg.d_model, max_len, cfg.pos_emb)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.N))
        self.dropout = nn.Dropout(cfg.p_drop)

    def forward(self, src, mask):
        x = self.dropout(self.pos(self.embedding(src)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, cfg, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(cfg.d_model, max_len, cfg.pos_emb)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.N))
        self.dropout = nn.Dropout(cfg.p_drop)

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.dropout(self.pos(self.embedding(tgt)))
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, cfg, max_len):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(vocab_size, cfg, max_len)
        self.decoder = Decoder(vocab_size, cfg, max_len)
        self.fc_out = nn.Linear(cfg.d_model, vocab_size)

    def encode(self, src):
        src_mask = (src != PAD).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, src_mask), src_mask

    def decode(self, tgt, enc_out, src_mask):
        pad_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device)).bool()
        return self.decoder(tgt, enc_out, pad_mask & causal.unsqueeze(0), src_mask)

    def forward(self, src, tgt):
        enc_out, src_mask = self.encode(src)
        return self.fc_out(self.decode(tgt, enc_out, src_mask))


class NoamScheduler(torch.optim.lr_scheduler.LambdaLR):
    """论文使用的 warmup + 反平方根学习率调度。"""

    def __init__(self, optimizer, d_model, warmup):
        self.d_model, self.warmup = d_model, warmup
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        step = max(step, 1)
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)


# ======================
# Data
# ======================
class WMT14Dataset(Dataset):
    """按需使用 SentencePiece 编码，避免一次性占用大量内存。"""

    def __init__(self, hf_dataset, sp, max_len):
        self.data = hf_dataset
        self.sp = sp
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]["translation"]
        src = [BOS] + self.sp.encode(item["de"], out_type=int)[: self.max_len - 2] + [EOS]
        tgt = [BOS] + self.sp.encode(item["en"], out_type=int)[: self.max_len - 2] + [EOS]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD),
        pad_sequence([t[:-1] for t in tgts], batch_first=True, padding_value=PAD),
        pad_sequence([t[1:] for t in tgts], batch_first=True, padding_value=PAD),
    )


def train_sentencepiece(train_data, model_path, vocab_size, max_samples, num_threads):
    """训练源语言和目标语言共享的 BPE 词表。"""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return

    count = min(max_samples, len(train_data))
    text_file = model_path.with_suffix(".corpus.txt")
    with text_file.open("w", encoding="utf-8") as f:
        for i in range(count):
            item = train_data[i]["translation"]
            f.write(item["de"].replace("\n", " ") + "\n")
            f.write(item["en"].replace("\n", " ") + "\n")

    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(model_path.with_suffix("")),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        hard_vocab_limit=False,
        pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK,
        num_threads=num_threads,
        minloglevel=1,
    )
    text_file.unlink(missing_ok=True)


# ======================
# Logging / Debug printing
# ======================
def log(message, variant=None):
    """带时间戳的日志输出，后台运行时会完整写入日志文件。"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"[{variant}] " if variant else ""
    print(f"[{stamp}] {tag}{message}", flush=True)


def decode_sentence(sp, ids):
    """把 token id 序列还原成句子，过滤掉特殊符号。"""
    tokens = ids.tolist() if hasattr(ids, "tolist") else ids
    return sp.decode([int(token) for token in tokens if int(token) not in (PAD, BOS, EOS)])


def print_first_batch(stage, variant, sp, src, tgt_out):
    """打印训练/验证时第一个样本的输入句子和目标句子。"""
    log(f"{stage} 第一个句子:", variant)
    print(f"  源语言输入: {decode_sentence(sp, src[0])}", flush=True)
    print(f"  源语言token: {src[0].tolist()}", flush=True)
    print(f"  目标语言标签: {decode_sentence(sp, tgt_out[0])}", flush=True)
    print(f"  目标语言token: {tgt_out[0].tolist()}\n", flush=True)


def print_first_test(variant, sp, src, reference_ids, hypothesis):
    """打印测试时第一个样本的输入、标准答案和模型输出。"""
    log("test 第一个句子:", variant)
    print(f"  源语言输入: {decode_sentence(sp, src)}", flush=True)
    print(f"  标准答案: {decode_sentence(sp, reference_ids)}", flush=True)
    print(f"  模型输出: {hypothesis}\n", flush=True)


def peek_first_pair(stage, variant, sp, loader):
    """从验证/测试 DataLoader 中窥视一个 batch，不消耗训练主循环。"""
    src, _, tgt_out = next(iter(loader))
    print_first_batch(stage, variant, sp, src, tgt_out)


@torch.no_grad()
def peek_first_test(variant, sp, model, loader, args, device):
    """用当前模型翻译测试集第一个句子，在每个 epoch 开始时输出。"""
    src, tgt_in, _ = next(iter(loader))
    hypothesis = beam_search(
        model, src[0], sp, args.max_len, args.beam_size,
        args.length_penalty, device,
    )
    reference_ids = [token for token in tgt_in[0].tolist() if token not in (PAD, BOS, EOS)]
    print_first_test(variant, sp, src[0], reference_ids, hypothesis)


# ======================
# Train / Evaluate
# ======================
@torch.no_grad()
def evaluate_ppl(model, loader, criterion, device):
    """计算验证集 PPL，并带进度条。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    # 用独立的 sum reduction 评估，避免修改训练用的 mean reduction。
    eval_criterion = nn.CrossEntropyLoss(
        ignore_index=PAD,
        label_smoothing=criterion.label_smoothing,
        reduction="sum",
    )
    for src, tgt_in, tgt_out in tqdm(loader, desc="eval ppl", leave=False, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        total_loss += eval_criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1)).item()
        total_tokens += (tgt_out != PAD).sum().item()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def beam_search(model, src, sp, max_len, beam_size, length_penalty, device):
    """简化版 beam search，长度惩罚与论文一致使用 alpha≈0.6。"""
    src = src.unsqueeze(0).to(device)
    enc_out, src_mask = model.encode(src)

    active = [(0.0, torch.tensor([[BOS]], device=device))]
    finished = []

    for _ in range(max_len - 1):
        candidates = []
        for score, seq in active:
            logits = model.fc_out(model.decode(seq, enc_out, src_mask))[:, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            top_scores, top_ids = log_probs.topk(beam_size)
            for i in range(beam_size):
                token = top_ids[0, i].view(1, 1)
                candidates.append(
                    (score + top_scores[0, i].item(), torch.cat([seq, token], dim=1))
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        next_active = []
        for score, seq in candidates:
            if seq[0, -1].item() == EOS:
                finished.append((score / (len(seq[0]) ** length_penalty), seq))
            else:
                next_active.append((score, seq))
        active = next_active[:beam_size]
        if not active:
            break

    pool = finished + [(score / (len(seq[0]) ** length_penalty), seq) for score, seq in active]
    if not pool:
        return ""
    best = max(pool, key=lambda item: item[0])[1][0].tolist()
    return sp.decode([token for token in best if token not in (PAD, BOS, EOS)])


@torch.no_grad()
def evaluate_bleu(model, loader, sp, args, device):
    """计算测试集 BLEU，并带进度条。"""
    model.eval()
    hypotheses, references = [], []
    for src, tgt_in, _ in tqdm(loader, desc="eval bleu", leave=False, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        for i in range(src.size(0)):
            hypothesis = beam_search(
                model, src[i], sp, args.max_len, args.beam_size,
                args.length_penalty, device,
            )
            reference_ids = [token for token in tgt_in[i].tolist() if token not in (PAD, BOS, EOS)]
            references.append([sp.decode(reference_ids)])
            hypotheses.append(hypothesis)

    import sacrebleu
    return sacrebleu.corpus_bleu(hypotheses, references).score


def train_and_evaluate(
    model, train_loader, val_loader, test_loader, criterion, optimizer,
    scheduler, sp, args, device, target_steps, ckpt_dir,
):
    """按 epoch 训练，并在每个 epoch 结束时评估、保存最优模型。"""
    step = 0
    best_ppl, best_bleu = float("inf"), 0.0
    best_epoch = best_step = 0
    best_ckpt = ckpt_dir / f"{args.variant}_best.pt"
    run_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if step >= target_steps:
            log(f"已达到目标步数 {target_steps}，提前结束训练", args.variant)
            break

        epoch_started = time.perf_counter()

        # batch 总数按 DataLoader 长度计算；若目标步数不足一轮，则按剩余步数截断。
        dataset_batches = len(train_loader)
        batches_this_epoch = min(dataset_batches, (target_steps - step) * args.accum_steps)
        # 只有攒满 accum_steps 个 batch 才会更新一次参数，因此这里用 floor。
        steps_this_epoch = batches_this_epoch // args.accum_steps
        log(
            f"epoch {epoch}/{args.epochs} 开始 | global_step={step}/{target_steps} | "
            f"dataset_batches={dataset_batches} batches_this_epoch={batches_this_epoch} "
            f"optimizer_steps={steps_this_epoch} accum_steps={args.accum_steps}",
            args.variant,
        )

        # 每个 epoch 开始时打印三个数据集的首个句子。
        model.eval()
        peek_first_pair("val", args.variant, sp, val_loader)
        peek_first_test(args.variant, sp, model, test_loader, args, device)
        model.train()

        progress = tqdm(
            total=steps_this_epoch,
            desc=f"epoch {epoch}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        accum, printed_train = 0, False
        batch_in_epoch = 0
        epoch_step = 0
        next_progress_pct = args.log_progress_percent
        optimizer.zero_grad(set_to_none=True)

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            if not printed_train:
                print_first_batch("train", args.variant, sp, src, tgt_out)
                printed_train = True

            logits = model(src, tgt_in)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            (loss / args.accum_steps).backward()
            accum += 1
            batch_in_epoch += 1

            if accum == args.accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum, step = 0, step + 1
                epoch_step += 1
                progress.update(1)
                progress.set_postfix(
                    batch=f"{batch_in_epoch}/{batches_this_epoch}",
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    time=datetime.now().strftime("%H:%M:%S"),
                )

                # 可选：按 epoch 百分比记录，默认关闭。
                if args.log_progress_percent > 0 and batches_this_epoch > 0:
                    current_pct = batch_in_epoch / batches_this_epoch * 100
                    while current_pct >= next_progress_pct and next_progress_pct <= 100:
                        log(
                            f"epoch {epoch} 进度 {next_progress_pct:.0f}% | "
                            f"batch={batch_in_epoch}/{batches_this_epoch} "
                            f"epoch_step={epoch_step}/{steps_this_epoch} "
                            f"global_step={step}/{target_steps} "
                            f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                            f"elapsed={time.perf_counter() - run_started:.0f}s",
                            args.variant,
                        )
                        next_progress_pct += args.log_progress_percent

                # 默认每 100 个 optimizer step 写一条日志。
                if args.log_interval > 0 and step % args.log_interval == 0:
                    log(
                        f"step {step}/{target_steps} loss={loss.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed={time.perf_counter() - run_started:.0f}s",
                        args.variant,
                    )

            # 可选：按 batch 记录，默认关闭。
            if args.log_batch_interval > 0 and batch_in_epoch % args.log_batch_interval == 0:
                log(
                    f"epoch {epoch} batch={batch_in_epoch}/{batches_this_epoch} | "
                    f"global_step={step}/{target_steps} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={time.perf_counter() - run_started:.0f}s",
                    args.variant,
                )

            if step >= target_steps:
                break

        progress.close()
        log(
            f"epoch {epoch} 训练结束 | epoch_batches={batch_in_epoch}/{batches_this_epoch} "
            f"epoch_steps={epoch_step}/{steps_this_epoch} "
            f"global_step={step}/{target_steps} "
            f"elapsed={time.perf_counter() - epoch_started:.0f}s",
            args.variant,
        )

        ppl = evaluate_ppl(model, val_loader, criterion, device)
        bleu = evaluate_bleu(model, test_loader, sp, args, device)
        log(f"epoch {epoch} 验证/测试 | PPL={ppl:.2f} BLEU={bleu:.1f}", args.variant)

        if ppl < best_ppl:
            previous = best_ppl
            best_ppl, best_bleu = ppl, bleu
            best_epoch, best_step = epoch, step
            torch.save(
                {
                    "config": asdict(model.cfg),
                    "state_dict": model.state_dict(),
                    "spm_path": str(ckpt_dir.parent / "spm" / f"de-en.bpe.{args.vocab_size}.model"),
                    "epoch": epoch,
                    "step": step,
                    "ppl": ppl,
                    "bleu": bleu,
                },
                best_ckpt,
            )
            log(
                f"最优模型更新 | PPL {previous:.2f} -> {ppl:.2f} "
                f"BLEU={bleu:.1f} epoch={epoch} step={step} | 已保存 {best_ckpt}",
                args.variant,
            )
        else:
            log(
                f"本轮未超过最优 PPL={best_ppl:.2f}，保留 epoch={best_epoch} 的 checkpoint",
                args.variant,
            )

    return {
        "actual_train_steps": step,
        "best_ppl": best_ppl,
        "best_bleu": best_bleu,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "checkpoint": str(best_ckpt) if best_epoch else "",
    }


# ======================
# Result saving
# ======================
RESULT_FIELDS = [
    "run_id", "variant", "paper_label", "timestamp",
    "N", "d_model", "d_ff", "h", "d_k", "d_v", "p_drop", "eps_ls", "pos_emb",
    "epochs", "actual_train_steps", "best_epoch", "best_step",
    "batch_size", "accum_steps", "vocab_size", "max_len",
    "params_m", "ppl_dev", "bleu_dev", "paper_ppl", "paper_bleu",
    "ppl_delta", "bleu_delta", "paper_params_m", "checkpoint",
]


def append_result(result, results_dir):
    csv_path = results_dir / "variations_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    return csv_path


def write_comparison(results_dir):
    """根据累计结果重新生成 Markdown 和 PNG 对比表。"""
    csv_path = results_dir / "variations_results.csv"
    if not csv_path.exists():
        return

    latest = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            latest[row["variant"]] = row

    rows = [latest[name] for name in VARIANT_ORDER if name in latest]
    if not rows:
        return

    md = [
        "| variant | paper | PPL (ours) | PPL (paper) | ΔPPL | BLEU (ours) | BLEU (paper) | ΔBLEU | params (ours/paper) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {variant} | {paper_label} | {ppl_dev} | {paper_ppl} | {ppl_delta} | "
            "{bleu_dev} | {paper_bleu} | {bleu_delta} | {params_m}/{paper_params_m} |".format(**row)
        )
    (results_dir / "variations_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["variant", "paper", "PPL ours", "PPL paper", "ΔPPL", "BLEU ours", "BLEU paper", "ΔBLEU"]
    cell_text = [
        [r["variant"], r["paper_label"], r["ppl_dev"], r["paper_ppl"],
         r["ppl_delta"], r["bleu_dev"], r["paper_bleu"], r["bleu_delta"]]
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1.2), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.get_text().set_weight("bold")
    fig.savefig(results_dir / "variations_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================
# Main
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANT_ORDER, help="本次只运行一个变体")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=9527)

    parser.add_argument("--spm-train-samples", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 表示使用全部训练数据")
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=3_000)
    parser.add_argument("--max-len", type=int, default=100)

    parser.add_argument("--train-steps", type=int, default=0, help="总训练步数上限；0 表示使用变体论文默认步数")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数；达到 train-steps 后会提前结束")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4_000)
    parser.add_argument(
        "--log-batch-interval", type=int, default=0,
        help="每处理 N 个 batch 写一条日志；默认关闭",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="每 N 个 optimizer step 写一条日志；默认 100",
    )
    parser.add_argument(
        "--log-progress-percent", type=float, default=0.0,
        help="按 epoch 百分比写日志；默认关闭，避免日志过密",
    )
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    return parser.parse_args()


def load_split(data_dir, split, max_samples):
    from datasets import load_dataset

    files = sorted(str(path) for path in data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 {data_dir}/{split}-*.parquet，请先运行 download_dataset.py")
    dataset = load_dataset("parquet", data_files={split: files}, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def main():
    args = parse_args()
    cfg = VARIANTS[args.variant]
    if args.train_steps:
        cfg = VariantConfig(**{**asdict(cfg), "train_steps": args.train_steps})

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    variant_results_dir = args.results_dir / args.variant
    variant_results_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_split(args.data_dir, "train", args.max_train_samples)
    val_data = load_split(args.data_dir, "validation", args.max_val_samples)
    test_data = load_split(args.data_dir, "test", args.max_test_samples)

    sp_path = variant_results_dir / "spm" / f"de-en.bpe.{args.vocab_size}.model"
    train_sentencepiece(
        train_data, sp_path, args.vocab_size,
        args.spm_train_samples, os.cpu_count() or 4,
    )

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    train_loader = DataLoader(
        WMT14Dataset(train_data, sp, args.max_len),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        WMT14Dataset(val_data, sp, args.max_len),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        WMT14Dataset(test_data, sp, args.max_len),
        batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model = Transformer(sp.get_piece_size(), cfg, args.max_len).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log(
        f"params={params_m:.1f}M epochs={args.epochs} "
        f"target_steps={cfg.train_steps} device={device}",
        args.variant,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.eps_ls)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg.d_model, args.warmup)

    ckpt_dir = variant_results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary = train_and_evaluate(
        model, train_loader, val_loader, test_loader, criterion, optimizer,
        scheduler, sp, args, device, cfg.train_steps, ckpt_dir,
    )

    ppl, bleu = summary["best_ppl"], summary["best_bleu"]
    log(
        f"训练完成 | best_epoch={summary['best_epoch']} "
        f"best_step={summary['best_step']}/{summary['actual_train_steps']} "
        f"PPL={ppl:.2f} BLEU={bleu:.1f} | {summary['checkpoint']}",
        args.variant,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "run_id": run_id,
        "variant": args.variant,
        "paper_label": cfg.paper_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "N": cfg.N, "d_model": cfg.d_model, "d_ff": cfg.d_ff,
        "h": cfg.h, "d_k": cfg.d_k, "d_v": cfg.d_v,
        "p_drop": cfg.p_drop, "eps_ls": cfg.eps_ls, "pos_emb": cfg.pos_emb,
        "epochs": args.epochs,
        "actual_train_steps": summary["actual_train_steps"],
        "best_epoch": summary["best_epoch"],
        "best_step": summary["best_step"],
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "vocab_size": sp.get_piece_size(), "max_len": args.max_len,
        "params_m": f"{params_m:.1f}",
        "ppl_dev": f"{ppl:.2f}", "bleu_dev": f"{bleu:.1f}",
        "paper_ppl": f"{cfg.paper_ppl:.2f}", "paper_bleu": f"{cfg.paper_bleu:.1f}",
        "ppl_delta": f"{ppl - cfg.paper_ppl:+.2f}",
        "bleu_delta": f"{bleu - cfg.paper_bleu:+.1f}",
        "paper_params_m": f"{cfg.paper_params:.1f}",
        "checkpoint": summary["checkpoint"],
    }
    append_result(result, variant_results_dir)
    write_comparison(variant_results_dir)
    log(f"结果已保存到 {variant_results_dir / 'variations_results.csv'}", args.variant)


if __name__ == "__main__":
    main()
