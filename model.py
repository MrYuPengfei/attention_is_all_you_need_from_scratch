# model.py
import torch
import torch.nn as nn
import math

# ======================
# Positional Encoding
# ======================
class PositionalEncoding(nn.Module):
    #目的(1, L, d_model)，先创(L, d_model)
    # (B, L, d_model) + (1, L, d_model) → (B, L, d_model)
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # position 扩维成 [max_len, 1]-->[[0],[1],[2],[3],..,[max_len-1]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 扩维成 [1, d_model/2]
        # torch.arange(0, d_model, 2)-->[0,2,4,..,d_model-2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # position * div_term 的大小是(max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 在第 0 维（最前面）新增一个维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # register_buffer：不会参与训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, d_model)，广播机制
        # 取前L个位置编码，广播加到batch里所有样本上
        x = x + self.pe[:, :x.size(1), :]
        return x

# ====================== 
# Scaled Dot-Product Attention
# ======================
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, heads, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = attn @ V
    return output

# ======================
# Multi-Head Attention
# ======================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        #如果 d_model % num_heads != 0，程序会抛出 AssertionError 异常并终止执行
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 四个投影矩阵：W_Q、W_K、W_V、W_O
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # (B, Lq, d_model) → (B, Lq, num_heads, d_k) → (B, num_heads, Lq, d_k)
        Q = self.W_Q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            # transpose(dim1, dim2)
            # 作用：**交换两个指定维度的顺序**，返回新张量，**共享原始内存，不会复制数据**，只是改张量的 shape、stride（步长）。
            # contiguous()
            # 作用：**返回一个连续的内存块**，返回的张量和原始张量共享内存，不会复制数据。
            # view(shape)
            # 作用：**改变张量的形状**，返回一个新的张量，**不改变原始张量**。 
            # 这里不是每个头单独一套 Linear，是一个大 Linear，之后用`view`切分，等价效果。
            # 拆解 Q 的变换（K、V 完全一样）：
            # 输入 Q 原始：`[B, Lq, d_model]`
            # 1. `self.W_Q(Q)`：线性投影，shape 不变：`[B, Lq, d_model]`
            # 2. `.view(B, -1, num_heads, d_k)`
            #    - `-1`自动填序列长度\(L_q\)
            #    - 变成：`[B, Lq, num_heads, d_k]`
            #    含义：每个位置上，拆成`num_heads`个小块，每块长度\(d_k\)
            # 3. `.transpose(1,2)` 交换第 1、第 2 维
            # 输出：`[B, num_heads, Lq, d_k]`
            # 维度含义：`[batch, 头编号, 序列位置, 单头维度]`
            # > 为什么要这么转置？
            # > 这么转置的目的：**把 “头” 放到 batch 那一维旁边，方便并行做注意力**，每一个头独立计算缩放点积。

        # Attention
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads (B, num_heads, L, d_k) → (B, L, num_heads, d_k) → (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
            # 1. `.transpose(1,2)`：把头和序列维换回来
            # `[B, num_heads, L, d_k]` → `[B, L, num_heads, d_k]`
            # 2. `.contiguous()`：**必须加！**
            # transpose 之后张量在内存里不连续，直接 view 会报错；contiguous 重新拷贝成连续内存。
            # 3. `.view(B, -1, num_heads * d_k)`
            # 把多个头在最后一维拼在一起：
            # `[B, L, num_heads, d_k]` → `[B, L, d_model]`
            # 因为 \(num\_heads \times d_k = d_{model}\)，回到原始输入维度
            # 4. `self.W_O(out)`：拼接后的多头特征再过一层线性层，做信息融合，得到最终多头注意力输出。
        out = self.W_O(out)
        return out
 



# ======================
# Feed Forward
# ======================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

# ======================
# Encoder Layer
# ======================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.mha(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ff(x)))         
        # x2 = self.norm1(x + self.dropout(self.mha(x, x, x, mask)))
        # x3 = self.norm2(x2 + self.dropout(self.ff(x2)))
        # return x3
        # 这两种写法的差异：### 差异点只有变量名：
        # - 写法 1：原地复用变量 `x`，覆盖原来的 x；
        # - 写法 2：用新变量 `x2, x3`，保留每一步中间结果，不覆盖。
        # > 
        # > PyTorch 里张量赋值只是绑定引用，**原地覆盖不会影响反向传播**。
        # > 因为这里 `x = ...` 是重新绑定变量，不是 `x[:] = ...` 这种张量 in-place 修改，梯度计算完全一样
        # 在 Python + PyTorch 里：**变量名只是标签，不是张量本身**。
        # `a = tensor` 只是把标签 `a` 绑到那块张量数据上，不是复制一份张量。


# ======================
# Decoder Layer
# ======================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.enc_attn(x, enc_out, enc_out, memory_mask)))
        return  self.norm3(x + self.dropout(self.ff(x)))
        # x2 = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # x3 = self.norm2(x2 + self.dropout(self.enc_attn(x2, enc_out, enc_out, memory_mask)))
        # x4 = self.norm3(x3 + self.dropout(self.ff(x3)))
        # return x4

# ======================
# Encoder
# ======================
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_heads, max_len=512, dropout=0.1, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ======================
# Decoder
# ======================
class Decoder(nn.Module):
    def __init__(self, output_size, d_model, num_layers, num_heads, max_len=512, dropout=0.1, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

# ======================
# Mask functions
# ======================
def create_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
# 1. `seq != pad_idx` -> 得到 bool 张量 `[B, L]`
# 2. `.unsqueeze(1)` → 升维成 `[B, 1, L]`
# 3. `.unsqueeze(2)` → 升维成 `[B, 1, 1, L]`

def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask  # (L,L)
    # - `B`：batch
    # - `L_src`：源序列长度（encoder 输入）
    # - `L_tgt`：目标序列长度（decoder 输入）
    # - `pad_idx`：padding 填充符的 id，一般是 0
    # - mask 是**bool 张量**：`True = 允许看到，False = 屏蔽（mask掉）`
    # 1. `torch.ones(size, size)` 生成 `[L, L]` 全 1 矩阵
    # 2. `torch.tril()`：**保留下三角（含对角线），上三角全部置 0**
    # 3. `.bool()` 转布尔 mask


# ======================
# Transformer
# ======================
class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=512, num_layers=6, num_heads=8, max_len=512, dropout=0.1, padding_idx=0):
        super().__init__()
        self.encoder = Encoder(input_size, d_model, num_layers, num_heads, max_len, dropout, padding_idx)
        self.decoder = Decoder(output_size, d_model, num_layers, num_heads, max_len, dropout, padding_idx)
        self.fc_out = nn.Linear(d_model, output_size)
        self.padding_idx = padding_idx
        self.max_len = max_len

    def forward(self, src, tgt):
        # masks
        src_mask = create_pad_mask(src, self.padding_idx).to(src.device)
        tgt_mask = create_pad_mask(tgt, self.padding_idx).to(tgt.device)
        look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        combined_mask = tgt_mask & look_ahead_mask.unsqueeze(0).unsqueeze(0)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, combined_mask, src_mask)
        out = self.fc_out(dec_out)
        return out



if __name__ == "__main__":
    from data import cmn_words, eng_words, seq_len, pad

    batch_size, input_size , output_size = 32, len(cmn_words) + 3, len(eng_words) + 3
    model = Transformer(input_size, output_size, max_len=seq_len, padding_idx=pad)

    inputs = torch.randint(input_size, (batch_size, seq_len))
    outputs = torch.randint(output_size, (batch_size, seq_len))
    assert model(inputs, outputs).argmax(dim=-1).shape == outputs.shape
    print(model)
