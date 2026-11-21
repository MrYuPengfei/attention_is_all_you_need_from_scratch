"""复现《Attention Is All You Need》论文 Table 3 的模型变体对比表。

表格共 13 列：第一列是变体名/分组，其余列是模型超参数与实验指标。
运行后默认在项目根目录生成 variations_table.png。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能保存图片
import matplotlib.pyplot as plt

# 表头第一列留空：具体变体名（base、(A)、big 等）放在 ROWS 的第一列。
# 其余列用 matplotlib 的 mathtext 渲染下标和乘号，避免系统字体缺字。
HEADERS = [
    "",
    "N",
    r"$d_{model}$",
    r"$d_{ff}$",
    r"$h$",
    r"$d_k$",
    r"$d_v$",
    r"$P_{drop}$",
    r"$\epsilon_{ls}$",
    "train steps",
    "PPL (dev)",
    "BLEU (dev)",
    r"params $\times 10^{6}$",
]

# 表格数据：与论文 Table 3 保持一致。
# "—" 表示该行沿用 base 配置或未报告该项；(E) 行第二列是变体说明而非 N。
ROWS = [
    ["base", "6", "512", "2048", "8", "64", "64", "0.1", "0.1", "100K", "4.92", "25.8", "65"],
    ["(A)", "—", "—", "—", "1", "512", "512", "—", "—", "—", "5.29", "24.9", "—"],
    ["(A)", "—", "—", "—", "4", "128", "128", "—", "—", "—", "5.00", "25.5", "—"],
    ["(A)", "—", "—", "—", "16", "32", "32", "—", "—", "—", "4.91", "25.8", "—"],
    ["(A)", "—", "—", "—", "32", "16", "16", "—", "—", "—", "5.01", "25.4", "—"],
    ["(B)", "—", "—", "—", "—", "16", "—", "—", "—", "—", "5.16", "25.1", "58"],
    ["(B)", "—", "—", "—", "—", "32", "—", "—", "—", "—", "5.01", "25.4", "60"],
    ["(C)", "2", "—", "—", "—", "—", "—", "—", "—", "—", "6.11", "23.7", "36"],
    ["(C)", "4", "—", "—", "—", "—", "—", "—", "—", "—", "5.19", "25.3", "50"],
    ["(C)", "8", "—", "—", "—", "—", "—", "—", "—", "—", "4.88", "25.5", "80"],
    ["(C)", "—", "256", "—", "—", "32", "32", "—", "—", "—", "5.75", "24.5", "28"],
    ["(C)", "—", "1024", "—", "—", "128", "128", "—", "—", "—", "4.66", "26.0", "168"],
    ["(C)", "—", "—", "1024", "—", "—", "—", "—", "—", "—", "5.12", "25.4", "53"],
    ["(C)", "—", "—", "4096", "—", "—", "—", "—", "—", "—", "4.75", "26.2", "90"],
    ["(D)", "—", "—", "—", "—", "—", "—", "0.0", "—", "—", "5.77", "24.6", "—"],
    ["(D)", "—", "—", "—", "—", "—", "—", "0.2", "—", "—", "4.95", "25.5", "—"],
    ["(D)", "—", "—", "—", "—", "—", "—", "—", "0.0", "—", "4.67", "25.3", "—"],
    ["(D)", "—", "—", "—", "—", "—", "—", "—", "0.2", "—", "5.47", "25.7", "—"],
    [
        "(E)",
        "positional embedding",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "4.92",
        "25.7",
        "—",
    ],
    ["big", "6", "1024", "4096", "16", "—", "—", "0.3", "—", "300K", "4.33", "26.4", "213"],
]


def make_table(output: Path) -> None:
    """绘制表格并保存到 output。

    步骤：创建画布 -> 用 ax.table 排版 -> 统一边框/字体样式 -> 保存 PNG。
    """
    # figsize 需要足够宽，否则 (E) 行的 "positional embedding" 会被挤压截断。
    fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
    ax.axis("off")  # 表格已经承载全部信息，不需要坐标轴

    # 给内容较长的列（如第二列）分配更大宽度，再归一化到 [0, 1]。
    widths = [1.2, 2.2, 0.9, 0.8, 0.6, 0.7, 0.7, 0.9, 0.8, 1.1, 1.1, 1.1, 1.2]
    widths = [width / sum(widths) for width in widths]

    table = ax.table(
        cellText=ROWS,
        colLabels=HEADERS,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    # 固定字号，避免 matplotlib 因列宽自动缩小到难以阅读。
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)  # 只拉高行距，不改变已经计算好的列宽

    # 统一边框和表头样式；首列左对齐，便于阅读。
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.get_text().set_weight("bold")
        if col == 0:
            cell.get_text().set_ha("left")
            if row > 0 and ROWS[row - 1][0] in {"base", "big"}:
                cell.get_text().set_weight("bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"表格已保存: {output}")


def parse_args():
    """允许通过 --output 覆盖默认图片路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "variations_table.png",
        help="输出图片路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_table(parse_args().output)
