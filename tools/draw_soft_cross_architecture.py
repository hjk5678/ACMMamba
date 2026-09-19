"""Draw the once-per-stage Soft Cross model from its current configuration.

Run: python tools/draw_soft_cross_architecture.py
Outputs are editable SVG and high-resolution PNG. No model weights are loaded.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
CONFIG = ROOT / "configs/ablations/train_kust4k_encoder_cccc_soft_once_2242_150e.yaml"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                     "svg.fonttype": "none", "axes.unicode_minus": False})
INK, MUTED = "#203047", "#63758A"
A = ("#EDF4FE", "#477EB8")
B = ("#FFF2E5", "#B77A35")
SOFT = ("#FFF0EE", "#C65F55")
AS6 = ("#EAF5F7", "#348995")
FUSE = ("#EAF5ED", "#508960")
DEC = ("#F1EDFA", "#8570AC")
GRAY = ("#F6F8FB", "#B1BDCB")


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=12, color=INK, bold=False, ha="center"):
    return ax.text(x, y, label, ha=ha, va="center", fontsize=size,
                   color=color, weight="bold" if bold else "normal",
                   linespacing=1.45, zorder=5)


def rect(ax, x, y, w, h, palette=GRAY, label="", size=12, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0,rounding_size=9", facecolor=palette[0],
        edgecolor=palette[1], lw=1.4, zorder=3))
    if label:
        text(ax, x, y, label, size, bold=bold)


def group(ax, x, y, w, h, title, palette=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=12", facecolor="#FCFDFE",
        edgecolor=palette[1], lw=1.2, zorder=0))
    text(ax, x+18, y+23, title, 12, bold=True, ha="left")


def edge(ax, pts, color=MUTED, dashed=False, both=False):
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=1.5, ls="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1],
        arrowstyle="<->" if both else "-|>", mutation_scale=13,
        lw=1.5, color=color, ls="--" if dashed else "-", zorder=2))


def dot(ax, x, y, color):
    ax.add_patch(Circle((x, y), 3, facecolor=color, edgecolor="none", zorder=5))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Catch canvas clipping; all plotted labels must remain inside the figure.
    frame = fig.bbox
    for ax in fig.axes:
        for artist in ax.texts:
            bb = artist.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > frame.x1+1 or bb.y1 > frame.y1+1:
                raise ValueError(f"Text outside canvas: {artist.get_text()}")
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def overview(cfg):
    rhdb = cfg.get("decoder_type", "unet") == "rhdb"
    rhdb_stages = cfg.get("rhdb_options", {}).get("hypergraph_stages", [1, 2]) if rhdb else []
    fig, ax = canvas(1820, 1570)
    title = "ACMMamba · Soft Cross + RHDB" if rhdb else "ACMMamba · Soft Cross"
    text(ax, 65, 48, title, 27, bold=True, ha="left")
    text(ax, 65, 95, "每阶段一次软交互  /  One Soft Cross block per stage", 15, MUTED, ha="left")
    decoder_label = "Deep RHDB / shallow ResNet decoder" if rhdb else "ResNet decoder"
    text(ax, 65, 130, f"Depths [2, 2, 4, 2]     •     AS6 directional modeling     •     MLFM skip fusion     •     {decoder_label}", 12, MUTED, ha="left")
    text(ax, 455, 185, "双分支层次编码器 / Dual-branch encoder", 16, bold=True)
    text(ax, 1130, 185, "多尺度融合 / Fusion", 16, bold=True)
    text(ax, 1540, 185, "混合解码器 / Decoder" if rhdb else "残差解码器 / Decoder", 16, bold=True)

    for x, palette, title in [(235, A, "Modality A · RGB"), (685, B, "Modality B · SAR / TIR / GPS")]:
        rect(ax, x, 240, 310, 58, palette, title, 12, True)
        rect(ax, x, 313, 310, 58, palette, "Patch Embedding\nConv 3×3, stride 2  →  Conv 3×3, stride 2", 10.5)
        edge(ax, [(x,269),(x,284)], palette[1])
        edge(ax, [(x,342),(x,401)], palette[1])

    ys = [450, 710, 970, 1230]
    for i, (y, c, depth, scale) in enumerate(zip(ys, cfg["dims"], cfg["depths"], [4,8,16,32]), 1):
        group(ax, 65, y-88, 795, 208, f"Stage {i}")
        text(ax, 455, y-65, f"C={c}    H/{scale} × W/{scale}    depth={depth}", 11, bold=True)
        for x, palette, branch in [(235, A, "A"), (685, B, "B")]:
            rect(ax, x, y-18, 290, 62, palette,
                 f"{branch} · first block\n4 directional AS6 + residual / FFN", 11)
            rect(ax, x, y+65, 290, 50, palette,
                 f"{branch} · Self-AS6 block × {depth-1}", 11, True)
            edge(ax, [(x,y+13),(x,y+40)], palette[1])
            if rhdb and i == 4:
                end_y = y+103 if branch == "A" else y+114
                ax.plot([x,x],[y+90,end_y],color=palette[1],lw=1.5,zorder=1)
            else:
                edge(ax, [(x,y+90),(x,y+140)], palette[1])
        edge(ax, [(380,y-18),(540,y-18)], SOFT[1], dashed=True, both=True)
        text(ax, 460, y-43, "Soft Cross", 12, SOFT[1], True)
        text(ax, 460, y+5, "before AS6", 10, SOFT[1])
        text(ax, 460, y+63, "Self routing", 10, MUTED)

        # Both stage-end features branch to MLFM and continue down their encoders.
        for x, port_y, bus_x, dest_y, palette in [
            (235,y+103,905,y+3,A), (685,y+114,930,y+27,B)]:
            dot(ax,x,port_y,palette[1])
            edge(ax, [(x,port_y),(bus_x,port_y),(bus_x,dest_y),(1010,dest_y)], palette[1])
        rect(ax, 1130, y+15, 240, 94, FUSE, f"MLFM {i}\nAdd / Cat branches\nS{i}: {c} × H/{scale} × W/{scale}", 12, True)
        edge(ax, [(1250,y+15),(1390,y+15)], FUSE[1])
        text(ax, 1320, y-7, f"S{i} skip", 11, FUSE[1])
        if 5-i in rhdb_stages:
            rect(ax,1540,y+15,300,100,AS6,
                 f"Decoder {5-i} · RHDB\nResNet ×1  ∥  Region Hypergraph\nGated increments · {c} channels",11.5,True)
        else:
            rect(ax,1540,y+15,300,100,DEC,
                 f"Decoder {5-i} · ResNet\nBilinear ↑ + 1×1 Conv + Concat\n3 ResNet blocks · {c} channels",12,True)
        if i < 4:
            for x, palette in [(235,A),(685,B)]:
                rect(ax,x,y+158,290,36,palette,"Patch Merging · ↓2",11)
                edge(ax,[(x,y+176),(x,y+260-49)],palette[1])
            edge(ax,[(1540,y+260-35),(1540,y+65)],DEC[1])

    # Last modality outputs are not downsampled after Stage 4.
    if not rhdb:
        text(ax, 235, 1384, "A4", 11, A[1])
        text(ax, 685, 1384, "B4", 11, B[1])
    rect(ax,1340,1410,700,80,DEC,
         "Bottleneck: Conv 3×3, stride 2 → GN → GELU → ConvBlock\nS4′: 768 × H/64 × W/64",12,True)
    edge(ax,[(1130,1292),(1130,1370)],FUSE[1])
    edge(ax,[(1540,1370),(1540,1295)],DEC[1])
    if rhdb:
        rect(ax,460,1410,790,80,AS6)
        text(ax,460,1390,"RHDB (Decoder 1–2): Up → Concat(skip) → Conv1×1 → F",11.5,bold=True)
        text(ax,460,1424,r"$Y=\mathrm{Post}(F+\Delta_{\mathrm{loc}}+g\odot\Delta_{\mathrm{hg}})$",16)
    rect(ax,1540,326,300,70,DEC,"Segmentation Head\nConvBlock → Conv 1×1 → K classes",12,True)
    edge(ax,[(1540,415),(1540,361)],DEC[1])
    rect(ax,1540,240,300,58,FUSE,"Output logits · N × K × H × W",12,True)
    edge(ax,[(1540,291),(1540,269)],DEC[1])
    text(ax,1740,280,"Bilinear ↑4",10,MUTED)
    text(ax,65,1478,"C→S  |  C→S  |  C→S→S→S  |  C→S     C = Soft Cross block; S = Self block",13,bold=True,ha="left")
    text(ax,65,1513,"10 dual-branch blocks · 80 independent AS6 instances · 16 learned exchange scalars · MLFM runs once at each stage end",11,MUTED,ha="left")
    text(ax,65,1543,"A/B features continue to the next encoder stage. S1–S4 feed the decoder. Spatial ratios assume H and W divisible by 64; batch omitted internally.",10,MUTED,ha="left")
    stem = "acmmamba_soft_cross_rhdb_architecture" if rhdb else "acmmamba_soft_cross_once_architecture"
    save(fig,stem)


def details():
    fig,ax=canvas(1900,1600)
    text(ax,65,48,"Soft Cross · AS6 · MLFM",27,bold=True,ha="left")
    text(ax,65,91,"首个编码 block 的内部机制 / Inside the first block of each encoder stage",14,MUTED,ha="left")
    group(ax,45,125,1810,650,"A   Soft Cross block: learned mixing before directional AS6",SOFT)

    # One explicit two-branch flow: preprocessing -> original scans -> mixture -> AS6 -> sum.
    for y,palette,name in [(245,A,"A"),(415,B,"B")]:
        rect(ax,150,y,150,64,palette,f"Input X{name}",12,True)
        rect(ax,375,y,240,82,palette,"LN → Conv 1×1\nDWConv 3×3 → SiLU",11)
        rect(ax,615,y,180,82,palette,f"4-way scan\n[{name}1, {name}2, {name}3, {name}4]",11)
        edge(ax,[(225,y),(255,y)],palette[1])
        edge(ax,[(495,y),(525,y)],palette[1])
        edge(ax,[(705,y),(780,y)],palette[1])
        rect(ax,1220,y,240,92,AS6,f"AS6{name},1 … AS6{name},4\n4 independent processors\n(one per routed sequence)",11,True)
        rect(ax,1510,y,280,92,AS6,"Inverse directional mapping\nSpatial Adaptor per path\nElementwise sum of 4 maps",11)
        edge(ax,[(1050,y),(1100,y)],palette[1])
        edge(ax,[(1340,y),(1370,y)],AS6[1])
        rect(ax,1740,y,140,92,palette,"Conv 1×1\nLayerScale\nDropPath",11)
        edge(ax,[(1650,y),(1670,y)],palette[1])
    rect(ax,915,330,270,290,SOFT)
    text(ax,915,211,"SOFT CROSS",16,SOFT[1],True)
    text(ax,915,264,r"$A'=[A_1,A_2,\widetilde{A}_3,\widetilde{A}_4]$",14)
    text(ax,915,310,r"$B'=[B_1,B_2,\widetilde{B}_3,\widetilde{B}_4]$",14)
    text(ax,915,370,"Mix reverse paths only\nOriginal A/B paths used\nfor both destinations",11)
    text(ax,915,440,"4 trainable scalar logits",11,SOFT[1],True)
    text(ax,600,175,"1: row   2: column   3: reversed row   4: reversed column",10,MUTED)

    rect(ax,520,582,875,134,SOFT)
    text(ax,520,543,r"$\widetilde{A}_{d}=(1-\alpha_d)A_d+\alpha_d B_d$",18)
    text(ax,520,582,r"$\widetilde{B}_{d}=(1-\beta_d)B_d+\beta_d A_d$",18)
    text(ax,520,625,r"$d\in\{3,4\},\quad \alpha_d=\sigma(\theta^A_d),\quad \beta_d=\sigma(\theta^B_d)$",14)
    rect(ax,1420,582,790,134,DEC)
    text(ax,1420,543,"Residual updates, independently for A and B",12,bold=True)
    text(ax,1420,581,r"$U_m=X_m+\mathrm{DropPath}(\gamma_m\,\mathrm{Proj}(Z_m))$",15)
    text(ax,1420,623,r"$Y_m=U_m+\mathrm{DropPath}(\eta_m\,\mathrm{FFN}(\mathrm{LN}(U_m)))$",15)
    edge(ax,[(1740,461),(1740,515)],DEC[1])
    edge(ax,[(1810,245),(1835,245),(1835,555),(1815,555)],A[1])
    text(ax,80,690,"Initialization: α3 = α4 = β3 = β4 = 0.5. Learned per stage / destination branch / reverse direction; shared over samples, channels and positions.",11,MUTED,ha="left")
    text(ax,80,722,"The two forward paths remain unchanged. Exchange fractions approach self routing at 0 and hard exchange at 1; they are not input-dependent gates.",11,MUTED,ha="left")
    text(ax,80,754,"Remaining blocks in the stage re-scan updated A/B features using Self routing, while retaining information exchanged by the first block.",11,MUTED,ha="left")

    group(ax,45,800,1810,320,"B   AS6 path: selective state-space modeling with memory and spatial adaptation",AS6)
    xs=[160,385,615,865,1105,1360,1690]
    widths=[170,210,195,215,205,235,270]
    labels=["Routed sequence\nN × C × L","Conv1d 1×1\nsplit: u and gate","u → DWConv1d\n→ SiLU","Selective S6\ninput-derived Δ, B, C","Memory Adaptor\nhistory: 1, 4, 16, 64","LN × SiLU(gate)\n→ Conv1d 1×1","Inverse direction mapping\n→ Spatial Adaptor"]
    for j,(x,w,label) in enumerate(zip(xs,widths,labels)):
        rect(ax,x,919,w,94,AS6,label,11)
        if j: edge(ax,[(xs[j-1]+widths[j-1]/2,919),(x-w/2,919)],AS6[1])
    edge(ax,[(385,872),(385,852),(1360,852),(1360,872)],SOFT[1])
    text(ax,865,842,"gate branch",10,SOFT[1])
    text(ax,85,1007,"Memory Adaptor operates on S6 output sequences; Spatial Adaptor operates on restored 2-D maps.",12,MUTED,ha="left")
    text(ax,85,1050,"Spatial Adaptor: X + scale × PWConv 1×1(GELU(DWConv 3×3(X))). Summation across directions occurs after spatial adaptation.",11,MUTED,ha="left")
    text(ax,85,1089,"Each complete dual-branch block owns 4 AS6 paths for A and 4 for B. No AS6 parameter sharing across branches, directions or blocks.",11,MUTED,ha="left")

    group(ax,45,1145,1810,340,"C   MLFM: stage-end fusion for the decoder",FUSE)
    rect(ax,255,1236,320,64,A,"Final stage feature A → Conv 1×1",11,True)
    rect(ax,255,1351,320,64,B,"Final stage feature B → Conv 1×1",11,True)
    rect(ax,785,1236,330,70,FUSE,"Add → FusionConv\nFadd = φadd(A′ + B′)",12)
    rect(ax,785,1351,330,70,FUSE,"Concat → FusionConv\nFcat = φcat(Concat(A′, B′))",12)
    for pts,col in [([(415,1236),(620,1236)],A[1]),
                    ([(415,1351),(620,1351)],B[1]),
                    ([(470,1236),(495,1236),(495,1335),(620,1335)],A[1]),
                    ([(445,1351),(545,1351),(545,1252),(620,1252)],B[1])]: edge(ax,pts,col)
    dot(ax,470,1236,A[1]);dot(ax,445,1351,B[1])
    rect(ax,1350,1293,610,116,FUSE,"Si = ai · Fadd + bi · Fcat\nIndependent learned scalar weights; initial ai = bi = 0.5\nNo sigmoid or sum-to-one constraint",12,True)
    edge(ax,[(950,1236),(995,1236),(995,1270),(1045,1270)],FUSE[1])
    edge(ax,[(950,1351),(995,1351),(995,1316),(1045,1316)],FUSE[1])
    edge(ax,[(1655,1293),(1785,1293)],FUSE[1])
    text(ax,1730,1256,"Decoder",11,FUSE[1],True)
    text(ax,85,1425,"FusionConv = (Conv 3×3 → LN → GELU) × 2. Each scale owns a separate MLFM, applied only after all blocks in that stage.",11,MUTED,ha="left")
    text(ax,85,1460,"Soft Cross coefficients control reverse-path exchange; MLFM coefficients combine the Add/Cat outputs. These are different parameter sets.",11,MUTED,ha="left")
    text(ax,65,1526,"Verified against cross_mamba.py, as6.py, mlfm.py and the once_per_stage / depths=[2,2,4,2] configuration.",11,MUTED,ha="left")
    text(ax,65,1563,"AS6 = Adaptor-S6. LN = LayerNorm. PWConv = pointwise convolution. DWConv = depthwise convolution. Proj = Conv 1×1.",10,MUTED,ha="left")
    save(fig,"acmmamba_soft_cross_modules")


if __name__ == "__main__":
    config=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["model"]
    assert config["cross_mode"] == "soft"
    assert config["cross_frequency"] == "once_per_stage"
    assert config["stage_modes"] == ["cross"]*4
    assert config["depths"] == [2,2,4,2]
    assert config["dims"] == [96,192,384,768]
    assert config["decoder_type"] == "unet" and config["decoder_blocks_per_stage"] == 3
    assert config["fusion_mode"] == "mlfm" and config["soft_cross_init"] == 0.5
    overview(config)
    details()
    print(f"Saved architecture and module figures to {OUT}")
