-"""交互式德英翻译脚本。

加载 variations_train_val_test.py 训练得到的 checkpoint，
输入德语后输出模型生成的英语翻译。

用法：
    python interactive_de2en.py --checkpoint results/base/checkpoints/base_best.pt
    python interactive_de2en.py --checkpoint results/base/checkpoints/base_best.pt \
        --text "Das ist ein Test."
"""

import argparse
from pathlib import Path

import torch

from variations_train_val_test import (
    BOS,
    EOS,
    Transformer,
    VariantConfig,
    VARIANTS,
    beam_search,
    decode_sentence,
)

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的 best.pt 路径")
    parser.add_argument(
        "--variant", default="base", choices=VARIANTS,
        help="变体名，仅用于查找默认 BPE 词表路径",
    )
    parser.add_argument("--spm-model", type=Path, default=None, help="显式指定 SentencePiece 模型路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--max-len", type=int, default=0, help="0 表示使用 checkpoint 中保存的值")
    parser.add_argument("--text", default=None, help="给定时只翻译这一句，不进入交互模式")
    parser.add_argument("--show-tokens", action="store_true", help="打印输入 token id")
    return parser.parse_args()


def resolve_spm_path(args, checkpoint, vocab_size):
    """按优先级查找 BPE 词表：命令行 > checkpoint > 变体目录。"""
    candidates = []
    if args.spm_model:
        candidates.append(args.spm_model)
    if checkpoint.get("spm_path"):
        candidates.append(Path(checkpoint["spm_path"]))
    candidates.append(
        args.checkpoint.parent.parent / "spm" / f"de-en.bpe.{vocab_size}.model"
    )
    candidates.append(
        ROOT / "results" / args.variant / "spm" / f"de-en.bpe.{vocab_size}.model"
    )

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"没有找到 BPE 词表，尝试过: {[str(p) for p in candidates]}")


def load_model(args):
    """加载 checkpoint、词表、模型配置，并返回推理所需对象。"""
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    config_dict = checkpoint["config"]
    cfg = VariantConfig(**config_dict)
    vocab_size = checkpoint.get("vocab_size", 0)
    max_len = args.max_len or checkpoint.get("max_len", 100)

    spm_path = resolve_spm_path(args, checkpoint, vocab_size or 32768)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    vocab_size = vocab_size or sp.get_piece_size()

    model = Transformer(vocab_size, cfg, max_len).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    print("模型已加载")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  spm model  : {spm_path}")
    print(f"  variant    : {args.variant}")
    print(f"  vocab size : {vocab_size}")
    print(f"  max len    : {max_len}")
    print(f"  beam size  : {args.beam_size}")
    print(f"  device     : {args.device}")
    print("输入德语后回车翻译；输入 q / quit / exit 退出。\n")

    return model, sp, max_len


def translate(model, sp, text, args, max_len):
    """把德语文本编码为 token，再调用 beam search 生成英语。"""
    token_ids = [BOS] + sp.encode(text, out_type=int)[: max_len - 2] + [EOS]
    src = torch.tensor(token_ids, dtype=torch.long, device=args.device)

    if args.show_tokens:
        print(f"输入 token: {token_ids}")
        print(f"输入文本  : {decode_sentence(sp, src)}\n")

    return beam_search(
        model,
        src,
        sp,
        max_len,
        args.beam_size,
        args.length_penalty,
        args.device,
    )


def main():
    args = parse_args()
    model, sp, max_len = load_model(args)

    if args.text is not None:
        print(translate(model, sp, args.text, args, max_len))
        return

    while True:
        try:
            text = input("de> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if text.lower() in {"q", "quit", "exit"}:
            print("bye")
            break
        if not text:
            continue

        try:
            output = translate(model, sp, text, args, max_len)
            print(f"en> {output}\n")
        except Exception as error:
            print(f"翻译失败: {error}\n")


if __name__ == "__main__":
    main()
