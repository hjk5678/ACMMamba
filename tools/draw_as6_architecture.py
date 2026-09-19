"""Draw the original SCSC/AS6/MLFM/ResNet baseline, without decoder-specific dependencies.

Run: python tools/draw_as6_architecture.py
"""
from draw_architecture import (
    canvas, txt, box, arrow, save, BLUE, ORANGE, GREEN, PURPLE, GRAY, MUTED,
)
from matplotlib.patches import FancyBboxPatch

AS6 = ("#FFF3CB", "#B48A2A")


def overview(highlight_as6=True):
    decoder_color = PURPLE
    fig, ax = canvas(2460 if highlight_as6 else 1660, 1340)
    txt(ax, 65, 46, "ACMMamba  |  双模态遥感语义分割", 25, weight="bold", ha="left")
    txt(ax, 65, 89, "四方向 Adaptor-S6 · Self–Cross–Self–Cross 编码 · MLFM 融合 · ResNet 残差解码", 13, MUTED, ha="left")
    txt(ax, 427, 138, "双分支层级编码器", 16, weight="bold")
    txt(ax, 997, 138, "多尺度融合", 16, weight="bold")
    txt(ax, 1390, 138, "ResNet 残差解码器", 16, weight="bold", color=decoder_color[1])
    for x, name, sub, palette in [(250, "模态 A · RGB", "N × C_a × H × W", BLUE),
                                  (605, "模态 B · SAR / TIR / 轨迹", "N × C_b × H × W", ORANGE)]:
        box(ax, x, 200, 255, 66, name, sub, palette, size=12)
        box(ax, x, 293, 245, 70, "Patch Embedding", "两次 3×3 Conv，stride 2", palette)
        arrow(ax, [(x,233), (x,258)])
        arrow(ax, [(x,328), (x,365)])

    for i, (y, c, scale, mode) in enumerate(zip(
            [427,627,827,1027], [96,192,384,768], [4,8,16,32],
            ["Self","Cross","Self","Cross"]), 1):
        ax.add_patch(FancyBboxPatch((100,y-62),655,124,
            boxstyle="round,pad=0,rounding_size=12", facecolor="#FBFCFE",
            edgecolor="#CBD4E0", linewidth=1.2, zorder=0))
        txt(ax, 117, y-45, f"Stage {i}  /  {mode}Mamba × 1", 12, weight="bold", ha="left")
        if highlight_as6:
            for x, modality, palette in [(250,"A",BLUE),(605,"B",ORANGE)]:
                box(ax, x, y+1, 225, 68, "", colors=palette)
                txt(ax, x, y-18, f"{modality} 分支 · {mode} 路由", 11, weight="bold")
                for dx in (-81,-27,27,81):
                    box(ax, x+dx, y+14, 47, 25, "AS6", colors=AS6, size=9)
        else:
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
        arrow(ax, [(1092,y),(1230,y)], color=GREEN[1])
        txt(ax, 1162, y-19, f"S{i} · skip", 11, GREEN[1])
        d = 5-i
        box(ax, 1390, y, 320, 130, f"Decoder {d} → D{d}",
            f"双线性 ↑2 → Conv1×1 → Concat\nResNet 基本块 × 3\n{c} × H/{scale} × W/{scale}", decoder_color, size=12)
        if i < 4:
            mid = y+100
            for x, palette in [(250,BLUE),(605,ORANGE)]:
                arrow(ax, [(x,y+62),(x,mid-24)], color=palette[1])
                box(ax, x, mid, 225, 48, "Patch Merging · ↓2", colors=palette, size=11)
                arrow(ax, [(x,mid+24),(x,y+200-62)], color=palette[1])
            arrow(ax, [(1390,y+200-65),(1390,y+65)], color=decoder_color[1])
            txt(ax, 1450, y+100, f"D{d-1} · deep", 10, decoder_color[1])

    box(ax, 1175, 1175, 570, 82,
        "Bottleneck · Conv3×3 / s2 → GN → GELU",
        "ConvBlock → S4′：768 × H/64 × W/64", PURPLE, size=12)
    arrow(ax, [(997,1063),(997,1134)], color=GREEN[1])
    arrow(ax, [(1390,1134),(1390,1092)], color=decoder_color[1])
    txt(ax, 1470, 1113, "S4′ · deep", 10, decoder_color[1])
    box(ax, 1390, 285, 300, 78, "Segmentation Head", "ConvBlock → Conv1×1 → K 通道", PURPLE)
    arrow(ax, [(1390,362),(1390,324)], color=decoder_color[1])
    box(ax, 1390, 194, 300, 64, "输出 Logits", "N × K × H × W", GREEN)
    arrow(ax, [(1390,246),(1390,226)], color=PURPLE[1])
    txt(ax, 1550, 237, "双线性 ↑4", 10, MUTED)
    txt(ax, 65, 1253, "原始基线：decoder_type=unet，decoder_blocks_per_stage=3；dims=[96,192,384,768]，depths=[1,1,1,1]。", 10.5, MUTED, ha="left")
    txt(ax, 65, 1281, "每级包含 3 个 ResNet 基本块：首块调整拼接后的通道，后两块保持通道数；归一化使用 GroupNorm。", 10.5, MUTED, ha="left")
    txt(ax, 65, 1309, "特征尺寸省略 N；比例按 H、W 为 64 的倍数标注。实际插值对齐 skip 尺寸；类别数 K 与输入通道随数据集配置。", 10.5, MUTED, ha="left")
    if highlight_as6:
        as6_inset(ax)
    stem = "acmmamba_as6_resnet_architecture" if highlight_as6 else "acmmamba_resnet_architecture"
    save(fig, stem)


def as6_inset(ax):
    """Show directional AS6 paths and the exact sequence/spatial boundary."""
    ax.plot([1640,1640],[138,1215],color="#DCE3EB",lw=1.2)
    txt(ax,2025,138,"AS6 在编码器内部的位置",18,AS6[1],weight="bold")
    box(ax,2025,212,690,76,"A / B 特征 → 归一化与预处理 → 四方向展开",
        "LN2d → Conv1×1 → DWConv3×3 → SiLU → 行 / 列 / 反向行 / 反向列",GRAY,size=12)
    box(ax,2025,324,690,102,"Self / Cross 路由（以下展开接收分支 A）",
        "Self：a1, a2, a3, a4    |    Cross：a1, a2, b3, b4\nB 分支对称处理，使用 B 自己的参数",BLUE,size=12)
    arrow(ax,[(2025,250),(2025,273)])
    xs=[1740,1930,2120,2310]
    # A distributor fans routed direction sequences out to independent AS6s.
    ax.plot([1740,2310],[403,403],color=MUTED,lw=1.5,zorder=1)
    ax.plot([2025,2025],[375,403],color=MUTED,lw=1.5,zorder=1)
    for i,x in enumerate(xs,1):
        box(ax,x,450,165,62,f"AS6 A{i}",f"方向 {i} · 序列处理",AS6,size=13)
        arrow(ax,[(x,403),(x,419)],color=AS6[1])
        box(ax,x,548,165,76,"逆方向映射",
            "→ Spatial Adaptor\n属于对应 AS6",AS6,size=11)
        arrow(ax,[(x,481),(x,510)],color=AS6[1])
        ax.plot([x,x],[586,612],color=AS6[1],lw=1.5,zorder=1)
    ax.plot([1740,2310],[612,612],color=AS6[1],lw=1.5,zorder=1)
    box(ax,2025,659,690,60,"四方向逐元素求和 Σ → Conv1×1 → 缩放 / DropPath → 加输入",colors=PURPLE,size=12)
    arrow(ax,[(2025,612),(2025,629)],color=AS6[1])
    box(ax,2025,740,690,58,"LN2d → ConvFFN → 缩放 / DropPath → 再次残差相加",colors=PURPLE,size=12)
    arrow(ax,[(2025,689),(2025,711)],color=PURPLE[1])

    ax.plot([1680,2370],[808,808],color="#DCE3EB",lw=1)
    txt(ax,2025,846,"单个 AS6：序列建模 + Memory + Spatial",17,AS6[1],weight="bold")
    # First row runs left to right; the second returns right to left.
    box(ax,1740,914,150,68,"方向序列","N × C × L",GRAY,size=12)
    box(ax,1930,914,175,68,"Conv1×1","分为 u 与 gate",AS6,size=12)
    box(ax,2125,914,170,68,"DWConv1d","→ SiLU",AS6,size=12)
    box(ax,2315,914,170,68,"Selective S6","Δ / B / C 由 u 生成",AS6,size=12)
    arrow(ax,[(1815,914),(1842.5,914)])
    arrow(ax,[(2017.5,914),(2040,914)])
    arrow(ax,[(2210,914),(2230,914)])
    box(ax,2315,1030,170,78,"Memory Adaptor","历史偏移\n1, 4, 16, 64",AS6,size=11)
    box(ax,2095,1030,205,78,"LN × SiLU(gate)","序列归一化与门控",AS6,size=11)
    box(ax,1855,1030,190,78,"Conv1×1","输出方向序列",AS6,size=12)
    arrow(ax,[(2315,948),(2315,991)],color=AS6[1])
    arrow(ax,[(2230,1030),(2197.5,1030)],color=AS6[1])
    arrow(ax,[(1992.5,1030),(1950,1030)],color=AS6[1])
    arrow(ax,[(1930,948),(1930,975),(2095,975),(2095,991)],color=ORANGE[1])
    txt(ax,2010,961,"gate",9,ORANGE[1])
    box(ax,2025,1146,690,84,"逆方向映射 → Spatial Adaptor",
        "X + scale × PWConv1×1(GELU(DWConv3×3(X)))\n在二维恢复后执行；Memory 检索 S6 输出序列",AS6,size=12)
    arrow(ax,[(1855,1069),(1855,1104)],color=AS6[1])
    txt(ax,2025,1236,"主图每个 AS6 小框对应一条独立方向路径。",12,AS6[1])
    txt(ax,2025,1266,"每个 DualScanMambaBlock 含 A/B 各 4 个 AS6，共 8 个。",11,MUTED)
    txt(ax,2025,1296,"AS6 位于 Self / Cross 内部；此基线四阶段 depth=1，共 32 个。",11,MUTED)


if __name__ == "__main__":
    overview(highlight_as6=True)
    print("Created docs/figures/acmmamba_as6_resnet_architecture.{svg,png}")
