# inference.py
import torch
from model import Transformer
from data import cmn_words, eng_words, sos, eos, seq_len, pad, get_dataset

# ---------------------
# 加载模型
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取词表大小
_, test_loader = get_dataset()
input_size = len(cmn_words) + eos + 1
output_size = len(eng_words) + eos + 1

# 初始化模型结构
model = Transformer(
    input_size=input_size,
    output_size=output_size,
    max_len=seq_len,
    padding_idx=pad
).to(device)

# 加载权重
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# ---------------------
# 翻译函数
# ---------------------
@torch.no_grad()
def translate(model, src_sentence, max_len=seq_len):
    """
    src_sentence: 中文输入序列（索引表示的tensor）
    """
    src = torch.tensor(src_sentence, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.full((1, 1), sos, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model(src, y)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        y = torch.cat([y, next_token], dim=1)
        if next_token.item() == eos:
            break

    output_indices = [i - 3 for i in y.view(-1).tolist() if i > 2]
    return " ".join(eng_words[i] for i in output_indices)

# # ---------------------
# # 测试：从 test_loader 取一个样本
# # ---------------------
# for src, tgt in test_loader:
#     src = src[0].tolist()
#     tgt = [i - eos - 1 for i in tgt[0].tolist() if i > eos]
#
#     src_sentence = [i for i in src if i > 0]
#     output = translate(model, src_sentence)
#
#     print("source:", " ".join(cmn_words[i - eos - 1] for i in src_sentence if i > eos))
#     print("target:", " ".join(eng_words[i] for i in tgt))
#     print("output:", output)
#     break  # 只演示一个样本
# ---------------------
# 用户输入交互模式
# ---------------------
print("\n🚀 Transformer 翻译器已加载！")
print("输入中文词语（以空格分隔），我将尝试输出英文翻译。")
print("输入 q 退出。\n")

while True:
    sentence = input("你: ").strip()
    if sentence.lower() == "q":
        print("再见 👋")
        break

    # 分词匹配（假设 cmn_words 是一个 list）
    tokens = sentence.split()
    src_indices = []
    for w in tokens:
        if w in cmn_words:
            src_indices.append(cmn_words.index(w) + 3)
        else:
            print(f"⚠️ 未知词：{w}")
            continue

    if not src_indices:
        continue

    output = translate(model, src_indices)
    print("🗣️ 英文输出:", output)
    print()