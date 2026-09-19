"""Code-derived RHDB figures: increment fusion and region hypergraph internals."""
from pathlib import Path
import yaml
from draw_soft_cross_architecture import (
    canvas, text, rect, group, edge, dot, save, A, B, AS6, FUSE, DEC, GRAY,
    SOFT, INK, MUTED,
)

ROOT = Path(__file__).resolve().parents[1]


def main_diagram():
    fig, ax = canvas(1640, 1410)
    text(ax,65,48,"RHDB · Residual–Hypergraph Dual-Branch Decoder",24,bold=True,ha="left")
    text(ax,65,93,"残差–超图双分支解码模块 / Current increment-fusion implementation",14,MUTED,ha="left")
    text(ax,65,128,"Parallel local / region-hypergraph branches · Spatial gating · Single outer identity",12,MUTED,ha="left")

    rect(ax,430,197,430,72,A,"Deep feature D\nB × Cd × Hd × Wd",13,True)
    rect(ax,1180,197,430,72,B,"Skip feature S (from MLFM)\nB × Cs × H × W",13,True)
    rect(ax,430,292,430,62,A,"Bilinear resize to skip resolution\nalign_corners = False",12)
    edge(ax,[(430,233),(430,261)],A[1])
    rect(ax,805,377,470,65,GRAY,"Concat[Up(D), S] → Conv 1×1",14,True)
    edge(ax,[(430,323),(430,377),(570,377)],A[1])
    edge(ax,[(1180,233),(1180,377),(1040,377)],B[1])
    rect(ax,805,465,260,54,FUSE,"F · B × C × H × W",14,True)
    edge(ax,[(805,410),(805,438)])
    dot(ax,805,515,INK)
    edge(ax,[(805,492),(805,515),(420,515),(420,552)],A[1])
    edge(ax,[(805,515),(1190,515),(1190,552)],AS6[1])

    group(ax,235,550,370,152,"Local residual branch",A)
    rect(ax,420,637,330,88,A,"1 ResNet BasicBlock\nConv3×3 / GN / GELU\nConv3×3 / GN → +F → GELU",12)
    group(ax,935,550,510,152,"Region hypergraph branch",AS6)
    rect(ax,1190,637,470,88,AS6,"Region nodes → KNN hypergraph\nHGConv → learned γ\nReshape / resize → bias-free Conv1×1",12)

    rect(ax,420,794,420,98,A)
    text(ax,420,773,"Local increment",13,A[1],True)
    text(ax,420,816,r"$\Delta_{\mathrm{loc}}=\mathrm{ResBlock}(F)-F$",17)
    edge(ax,[(420,681),(420,745)],A[1])
    rect(ax,1190,794,470,98,AS6)
    text(ax,1190,773,"Hypergraph increment only",13,AS6[1],True)
    text(ax,1190,816,r"$\Delta_{\mathrm{hg}}=\mathrm{Restore}(\gamma\,\mathrm{HGConv}(X))$",16)
    edge(ax,[(1190,681),(1190,745)],AS6[1])
    text(ax,1560,873,"No node identity added",10.5,MUTED,ha="right")

    # F is subtracted from the complete local output and added once outside.
    edge(ax,[(675,465),(140,465),(140,794),(210,794)],FUSE[1])
    dot(ax,140,794,FUSE[1])
    text(ax,160,723,"F for subtraction",11,FUSE[1],ha="left")
    edge(ax,[(140,794),(140,1100),(775,1100)],FUSE[1])
    text(ax,160,1070,"Outer identity F · added once",12,FUSE[1],ha="left")

    rect(ax,805,975,990,136,SOFT)
    text(ax,805,930,"Spatial gate and increment fusion",14,SOFT[1],True)
    text(ax,805,973,r"$g=\sigma(\mathrm{Conv}_{1\times1}([\Delta_{\mathrm{loc}},\Delta_{\mathrm{hg}}]))$",18)
    text(ax,805,1017,r"$\Delta=\Delta_{\mathrm{loc}}+g\odot\Delta_{\mathrm{hg}}\quad (g\in\mathbb{R}^{B\times1\times H\times W})$",17)
    edge(ax,[(420,843),(420,907)],A[1])
    edge(ax,[(1190,843),(1190,907)],AS6[1])
    rect(ax,805,1100,60,48,FUSE,"+",21,True)
    edge(ax,[(805,1043),(805,1076)],SOFT[1])
    rect(ax,805,1190,680,76,DEC,"Post: Conv 3×3 → GroupNorm → GELU\nY = Post(F + Δloc + g ⊙ Δhg)",14,True)
    edge(ax,[(805,1124),(805,1152)],DEC[1])
    rect(ax,805,1290,430,58,FUSE,"Output Y · B × C × H × W",14,True)
    edge(ax,[(805,1228),(805,1261)],DEC[1])
    text(ax,65,1351,"Current gate mode: the local increment is unweighted. Gate weights initialize to 0, bias to −2; g initially equals sigmoid(−2) ≈ 0.1192.",10.5,MUTED,ha="left")
    text(ax,65,1383,"γ is an unconstrained learned scalar initialized to 0. Initially Δhg = 0 and Y = Post(ResBlock(F)), up to floating-point rounding.",10.5,MUTED,ha="left")
    save(fig,"rhdb_internal_architecture")


def hypergraph_diagram():
    fig,ax=canvas(1900,1370)
    text(ax,65,48,"RHDB · Region Hypergraph Branch",26,bold=True,ha="left")
    text(ax,65,92,"区域超图分支 / Region construction, sample-specific topology and feature propagation",14,MUTED,ha="left")

    group(ax,45,130,1810,245,"A   Region nodes and normalized center coordinates",AS6)
    nodes=[(175,180,"Input F\nB × C × H × W"),
           (435,230,"Conv 1×1\nC → Ch = 64"),
           (760,320,"Adaptive average pooling\nhg=min(H,16), wg=min(W,16)"),
           (1130,310,"Flatten + transpose\nX: B × N × Ch")]
    for j,(x,w,label) in enumerate(nodes):
        rect(ax,x,238,w,88,AS6,label,12,True)
        if j:
            px,pw,_=nodes[j-1]
            edge(ax,[(px+pw/2,238),(x-w/2,238)],AS6[1])
    rect(ax,1590,238,440,88,GRAY,"Region-center coordinates P\nP: N × 2; normalized to [0,1]",12)
    edge(ax,[(760,194),(760,178),(1590,178),(1590,194)],MUTED)
    text(ax,1000,169,"derived from pooled grid",10,MUTED)
    text(ax,85,327,"N = hg × wg ≤ 256. Nodes represent pooled regions, not individual pixels. X and P follow the same row-major order.",12,MUTED,ha="left")

    group(ax,45,405,1810,365,"B   KNN hypergraph construction (independently for each sample)",SOFT)
    rect(ax,350,536,530,125,SOFT)
    text(ax,350,494,"Semantic–spatial distance",13,SOFT[1],True)
    text(ax,350,537,r"$d_{ij}=0.7(1-\cos(x_i,x_j))+0.3\Vert p_i-p_j\Vert_2$",15)
    text(ax,350,579,"Inputs: X and P; self distance set to +∞",11,MUTED)
    rect(ax,940,536,430,125,SOFT,"KNN selection\nK′ = min(8, N−1) other nodes\nAdd the center explicitly",13,True)
    rect(ax,1530,536,520,125,SOFT,"Binary incidence matrix H\nB × N vertices × N hyperedges\nOne hyperedge per center",13,True)
    edge(ax,[(615,536),(725,536)],SOFT[1])
    edge(ax,[(1155,536),(1270,536)],SOFT[1])
    text(ax,85,642,"Each hyperedge contains its center plus up to 8 neighbors (9 members when N ≥ 9). Columns of H are hyperedges, not adjacency columns.",11.5,MUTED,ha="left")
    text(ax,85,687,"Hard top-k construction runs without gradients. Node features X retain gradients through the subsequent HG propagation path.",11.5,MUTED,ha="left")
    text(ax,85,732,"The distance coefficient 0.7 is fixed by configuration. The topology depends on the current sample; it is not a trainable soft assignment.",11.5,MUTED,ha="left")

    group(ax,45,800,1810,360,"C   HGConv and restoration of the hypergraph increment",AS6)
    rect(ax,385,929,590,136,AS6)
    text(ax,385,882,"Degree-normalized propagation (W = I)",13,AS6[1],True)
    text(ax,385,929,r"$\overline{X}=D_v^{-1/2}HD_e^{-1}H^{\mathsf{T}}D_v^{-1/2}X$",19)
    text(ax,385,976,"Inputs: differentiable X and detached incidence H",11,MUTED)
    rect(ax,865,929,260,136,AS6,"Linear(Ch → Ch)\nGELU\n× learned scalar γ",13,True)
    rect(ax,1260,929,360,136,AS6,"Reshape to region grid\nBilinear resize to H × W\nConv 1×1: Ch → C, no bias",12,True)
    rect(ax,1665,929,260,100,FUSE,"Δhg only\nB × C × H × W",14,True)
    edge(ax,[(680,929),(735,929)],AS6[1])
    edge(ax,[(995,929),(1080,929)],AS6[1])
    edge(ax,[(1440,929),(1535,929)],AS6[1])
    text(ax,85,1054,"Dv: vertex degrees (row sums of H). De: hyperedge degrees (column sums of H). Degree denominators use eps = 1e-6.",11.5,MUTED,ha="left")
    text(ax,85,1094,"Pooling, distances, propagation, HG Linear and γ scaling run in FP32. H and its transpose are used without materializing dense G.",11.5,MUTED,ha="left")
    text(ax,85,1133,"γ starts at 0; the restoration projection has no bias. No node identity X is added to γ·HGConv(X).",11.5,MUTED,ha="left")

    text(ax,65,1211,"Baseline placement (code execution order)",15,bold=True,ha="left")
    for j,(x,label,palette) in enumerate([
        (300,"decoder1 · RHDB\nH/32, C=768",AS6),
        (720,"decoder2 · RHDB\nH/16, C=384",AS6),
        (1140,"decoder3 · ResNet ×3\nH/8, C=192",DEC),
        (1560,"decoder4 · ResNet ×3\nH/4, C=96",DEC)]):
        rect(ax,x,1276,340,75,palette,label,12,True)
        if j:edge(ax,[(x-420+170,1276),(x-170,1276)],DEC[1])
    text(ax,65,1343,"Diagram follows the gate / increment baseline. Stage channels and 64-channel HG projections come from the current 2-stage RHDB configuration.",10.5,MUTED,ha="left")
    save(fig,"rhdb_hypergraph_detail")


if __name__ == "__main__":
    p=ROOT / "configs/train_korea_no_cloud_baseline_rhdb.yaml"
    cfg=yaml.safe_load(p.read_text(encoding="utf-8"))["model"]
    expected={"hypergraph_stages":[1,2],"use_hypergraph":True,"fusion_mode":"gate",
              "k":8,"node_grid":16,"alpha":0.7,"hypergraph_hidden_dim":64,
              "gamma_init":0.0,"num_residual_blocks":1,"dropout":0.0}
    assert cfg["decoder_type"] == "rhdb"
    assert cfg["dims"] == [96,192,384,768]
    for key,value in expected.items():
        assert cfg["rhdb_options"][key] == value, (key,cfg["rhdb_options"][key])
    main_diagram()
    hypergraph_diagram()
    print("Saved RHDB main and hypergraph detail PNG/SVG figures.")
