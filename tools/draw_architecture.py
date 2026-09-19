"""Draw code-derived ACMMamba architecture figures (requires matplotlib).

Run from any directory: python tools/draw_architecture.py
The diagram describes the default S-C-S-C / MLFM model in model.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                     "svg.fonttype": "none", "axes.unicode_minus": False})
INK = "#203047"
MUTED = "#65758B"
BLUE = ("#E9F2FF", "#4E83C4")
ORANGE = ("#FFF0E1", "#C9904B")
GREEN = ("#E6F5EE", "#529878")
PURPLE = ("#F0ECFA", "#9278B9")
GRAY = ("#F4F6F9", "#ACB8C8")


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, size=13, color=INK, weight="normal", ha="center"):
    s = s.translate(str.maketrans({"ₐ": "_a", "ᵦ": "_b", "ᵢ": "_i", "₁": "1", "₂": "2", "₃": "3", "₄": "4"}))
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", linespacing=1.55)


def box(ax, x, y, w, h, title, sub="", colors=GRAY, size=13):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle="round,pad=0,rounding_size=10", linewidth=1.3,
                 edgecolor=colors[1], facecolor=colors[0], zorder=3))
    title_y = y - 12 - 10 * sub.count("\n") if sub else y
    txt(ax, x, title_y, title, size, weight="bold")
    if sub:
        txt(ax, x, y+16, sub, size-2, MUTED)


def arrow(ax, pts, color=MUTED, dashed=False, both=False):
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=1.5, linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1],
                 arrowstyle="<->" if both else "-|>", mutation_scale=13,
                 linewidth=1.5, color=color, linestyle="--" if dashed else "-", zorder=2))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas(1660, 1290)
    txt(ax, 65, 46, "ACMMamba  |  双模态语义分割网络", 25, weight="bold", ha="left")
    txt(ax, 65, 89, "代码默认结构 · Self–Cross–Self–Cross · 多尺度 MLFM 融合 · 四级残差 U-Net 解码器", 13, MUTED, ha="left")
    txt(ax, 427, 138, "双分支层级编码器", 16, weight="bold")
    txt(ax, 997, 138, "多尺度融合", 16, weight="bold")
    txt(ax, 1390, 138, "U-Net 解码器", 16, weight="bold")
    for x, name, sub, palette in [(250, "模态 A", "N × Cₐ × H × W", BLUE),
                                  (605, "模态 B", "N × Cᵦ × H × W", ORANGE)]:
        box(ax, x, 200, 245, 66, name, sub, palette)
        box(ax, x, 293, 245, 70, "Patch Embedding", "两次 3×3 Conv，stride 2", palette)
        arrow(ax, [(x,233), (x,258)])
        arrow(ax, [(x,328), (x,365)])

    ys = [427, 627, 827, 1027]
    for i, (y, c, scale, mode) in enumerate(zip(ys, [96,192,384,768], [4,8,16,32], ["Self","Cross","Self","Cross"]), 1):
        ax.add_patch(FancyBboxPatch((100,y-62),655,124,
                     boxstyle="round,pad=0,rounding_size=12", facecolor="#FBFCFE",
                     edgecolor="#CBD4E0", linewidth=1.2, zorder=0))
        txt(ax, 117, y-45, f"Stage {i}  /  {mode}Mamba × 1", 12, weight="bold", ha="left")
        box(ax, 250, y+1, 225, 48, f"A 分支 · {mode}", colors=BLUE, size=12)
        box(ax, 605, y+1, 225, 48, f"B 分支 · {mode}", colors=ORANGE, size=12)
        if mode == "Cross":
            arrow(ax, [(370,y+1),(485,y+1)], color="#AB7865", dashed=True, both=True)
            txt(ax, 427, y+35, "交换反向序列", 10, MUTED)
        else:
            txt(ax, 427, y+1, "独立扫描", 10, MUTED)
        txt(ax, 427, y+52, f"每个分支：{c} × H/{scale} × W/{scale}", 11, MUTED)
        arrow(ax, [(755,y),(901,y)], color=GREEN[1])
        txt(ax, 827, y-19, f"(A{i}, B{i})", 11, MUTED)
        box(ax, 997, y, 190, 72, f"MLFM {i}", f"S{i}  ·  {c} 通道", GREEN)
        arrow(ax, [(1092,y),(1250,y)], color=GREEN[1])
        txt(ax, 1170, y-19, f"S{i} 跳跃连接", 10, GREEN[1])
        d = 5-i
        box(ax, 1390, y, 280, 92, f"Decoder {d}", f"{c} × H/{scale} × W/{scale}\nUpBlock · 3 个残差块", PURPLE, size=13)
        if i < 4:
            mid = y+100
            for x, palette in [(250,BLUE),(605,ORANGE)]:
                arrow(ax, [(x,y+62),(x,mid-24)], color=palette[1])
                box(ax, x, mid, 225, 48, "Patch Merging · ↓2", colors=palette, size=11)
                arrow(ax, [(x,mid+24),(x,y+200-62)], color=palette[1])
            arrow(ax, [(1390,y+200-46),(1390,y+46)], color=PURPLE[1])

    box(ax, 1175, 1165, 520, 82, "Bottleneck · 3×3 Conv / stride 2 + ConvBlock",
        "S4′：768 × H/64 × W/64", PURPLE, size=13)
    arrow(ax, [(997,1063),(997,1124)], color=GREEN[1])
    arrow(ax, [(1390,1124),(1390,1073)], color=PURPLE[1])
    box(ax, 1390, 291, 280, 78, "Segmentation Head", "ConvBlock → 1×1 Conv → K", PURPLE)
    arrow(ax, [(1390,381),(1390,330)], color=PURPLE[1])
    box(ax, 1390, 196, 280, 64, "输出 Logits", "N × K × H × W", GREEN)
    arrow(ax, [(1390,252),(1390,228)], color=PURPLE[1])
    txt(ax, 1550, 240, "双线性 ↑4", 10, MUTED)
    txt(ax, 65, 1243, "主图采用默认 dims=[96,192,384,768]、depths=[1,1,1,1]；Cₐ、Cᵦ、K 随数据集配置。", 11, MUTED, ha="left")
    txt(ax, 65, 1268, "图中省略 batch 维；空间比例按 H、W 可被 64 整除标注。各阶段融合输出送往解码器，A/B 分支继续进入下一编码阶段。", 10, MUTED, ha="left")
    save(fig, "acmmamba_architecture")


def panel(ax, y, letter, title):
    ax.plot([55,1545], [y-28,y-28], color="#DEE4ED", lw=1)
    txt(ax, 60, y, f"{letter}   {title}", 18, weight="bold", ha="left")


def chain(ax, xs, y, labels, widths, palette=GRAY, height=64, size=11):
    for j, (x,label,w) in enumerate(zip(xs,labels,widths)):
        box(ax,x,y,w,height,label,colors=palette,size=size)
        if j:
            arrow(ax, [(xs[j-1]+widths[j-1]/2,y),(x-w/2,y)])


def details():
    fig, ax = canvas(1600, 1510)
    txt(ax, 60, 43, "ACMMamba  |  核心模块细节", 25, weight="bold", ha="left")
    txt(ax, 60, 84, "依据 cross_mamba.py、as6.py、mlfm.py 与 unet_decoder.py 的实际前向传播绘制", 12, MUTED, ha="left")
    panel(ax, 139, "A", "DualScanMambaBlock：Self / Cross 路由与残差更新")
    chain(ax, [125,340,560,795,1035,1340], 218,
          ["A / B 输入", "LN₂d → 1×1 Conv\n→ DWConv 3×3 → SiLU", "四方向展开\n行 / 列 / 反向行 / 反向列", "Self / Cross\n路由表", "4 个独立 AS6\n每个模态各自持有", "逐方向逆映射\n→ Spatial Adaptor\n→ 四方向求和"],
          [130,225,175,180,200,265], height=84, size=10.5)
    box(ax, 365, 350, 585, 124, "Self：A ← [a₁, a₂, a₃, a₄]；B ← [b₁, b₂, b₃, b₄]",
        "Cross：A ← [a₁, a₂, b₃, b₄]；B ← [b₁, b₂, a₃, a₄]\n正向行、列保留；反向行、列交换", BLUE, size=11.5)
    box(ax, 1080, 350, 765, 124, "两次残差更新（A、B 分别计算）",
        "x′ = x + DropPath(γscan · Conv1×1(Σ 恢复后的方向特征))\nxout = x′ + DropPath(γffn · ConvFFN(LN₂d(x′)))", PURPLE, size=12)
    arrow(ax, [(1340,260),(1340,288)], color=PURPLE[1])
    txt(ax, 80, 443, "A/B 不共享参数。ConvFFN：1×1 升维 → DWConv 3×3 → GELU → Dropout → 1×1 降维 → Dropout。", 11, MUTED, ha="left")

    panel(ax, 505, "B", "Adaptor-S6：序列建模、历史记忆与空间恢复")
    chain(ax, [145,370,625,890,1135,1390], 594,
          ["方向序列\n[N, C, L]", "1×1 Conv1d\n分为 u 与 gate", "u → DWConv1d\n→ SiLU", "Selective S6\n参数 Δ、B、C 来自 u", "Memory Adaptor", "LayerNorm\n× SiLU(gate)\n→ 1×1 Conv1d"],
          [160,210,210,235,190,225], height=88, size=11)
    arrow(ax, [(370,550),(370,537),(1390,537),(1390,550)], color=ORANGE[1])
    txt(ax, 1260, 524, "gate 分支", 10, ORANGE[1])
    box(ax, 442, 746, 730, 144, "Memory Adaptor · 在 S6 输出序列上检索历史",
        "历史偏移 {1, 4, 16, 64} → 与 query 计算相关性\n正 / 负相关分别 ReLU + Softmax，再取均值作为权重\ny = x + sigmoid(gate(x)) × 加权历史特征", GREEN, size=12)
    box(ax, 1185, 746, 625, 144, "逆方向映射 → Spatial Adaptor",
        "序列恢复为 [N, C, H, W] 后逐方向计算\ny = x + scale × PWConv1×1(GELU(DWConv3×3(x)))\n各方向恢复完成后，逐元素求和", PURPLE, size=12)
    arrow(ax, [(1390,638),(1390,674)], color=PURPLE[1])
    txt(ax, 80, 848, "默认 d_state=16，expand=1；Memory Adaptor 使用 S6 的输出序列，Spatial Adaptor 在逆映射之后执行。", 11, MUTED, ha="left")

    panel(ax, 910, "C", "MLFM：可学习的 Add / Cat 双分支融合")
    box(ax, 200, 1000, 255, 72, "Aᵢ → 1×1 Conv", "对齐至 Cᵢ 通道", BLUE)
    box(ax, 200, 1110, 255, 72, "Bᵢ → 1×1 Conv", "对齐至 Cᵢ 通道", ORANGE)
    box(ax, 610, 1000, 330, 72, "Add → FusionConv", "Fadd = Conv(A′ + B′)", GREEN)
    box(ax, 610, 1110, 330, 72, "Concat → FusionConv", "Fcat = Conv(Cat(A′, B′))", GREEN)
    arrow(ax, [(327.5,1000),(445,1000)], color=BLUE[1])
    arrow(ax, [(327.5,1110),(445,1110)], color=ORANGE[1])
    arrow(ax, [(350,1000),(378,1000),(378,1088),(445,1088)], color=BLUE[1])
    arrow(ax, [(350,1110),(403,1110),(403,1022),(445,1022)], color=ORANGE[1])
    box(ax, 1115, 1055, 480, 105, "Sᵢ = aᵢ · Fadd + bᵢ · Fcat",
        "aᵢ、bᵢ 独立可学习；初始值均为 0.5\n不施加归一化或和为 1 的约束", GREEN, size=13)
    arrow(ax, [(775,1000),(830,1000),(830,1033),(875,1033)], color=GREEN[1])
    arrow(ax, [(775,1110),(830,1110),(830,1077),(875,1077)], color=GREEN[1])
    arrow(ax, [(1355,1055),(1510,1055)], color=GREEN[1])
    txt(ax, 1450, 1030, "送入解码器", 11, GREEN[1])
    txt(ax, 80, 1178, "FusionConv = (3×3 Conv → LN₂d → GELU) × 2；每个尺度拥有独立的 MLFM 参数。", 11, MUTED, ha="left")

    panel(ax, 1240, "D", "UpBlock：逐级上采样与残差细化")
    chain(ax, [180,465,745,1030,1370], 1320,
          ["上一解码特征", "双线性插值\n对齐 skip 尺寸", "1×1 Conv\n调整通道", "Concat(skip)", "ResidualStage\n3 个 ResidualBlock"],
          [210,220,205,215,280], palette=PURPLE, height=76, size=12)
    arrow(ax, [(1030,1385),(1030,1358)], color=GREEN[1])
    txt(ax, 1030, 1405, "S4 / S3 / S2 / S1", 11, GREEN[1])
    txt(ax, 80, 1450, "ResidualBlock：Conv3×3 → GroupNorm → GELU → Dropout → Conv3×3 → GroupNorm → 加 shortcut → GELU。", 10.5, MUTED, ha="left")
    txt(ax, 80, 1479, "首个残差块用 1×1 Conv + GroupNorm 将拼接后的 shortcut 投影至目标通道，其余块保持通道数。", 10.5, MUTED, ha="left")
    save(fig, "acmmamba_modules")


if __name__ == "__main__":
    overview()
    details()
    print(f"Figures written to {OUT}")
