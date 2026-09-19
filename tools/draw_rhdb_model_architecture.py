"""Draw the configured Soft-Cross-once / MLFM / two-stage RHDB model."""
from pathlib import Path
import yaml
from draw_soft_cross_architecture import overview

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train_korea_no_cloud_baseline_rhdb.yaml"

if __name__ == "__main__":
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["model"]
    expected = {"decoder_type":"rhdb", "dims":[96,192,384,768],
                "depths":[2,2,4,2], "stage_modes":["cross"]*4,
                "cross_frequency":"once_per_stage", "cross_mode":"soft",
                "fusion_mode":"mlfm", "decoder_blocks_per_stage":3}
    for key,value in expected.items():
        assert cfg[key] == value, (key,cfg[key])
    assert cfg["rhdb_options"]["hypergraph_stages"] == [1,2]
    assert cfg["rhdb_options"]["num_residual_blocks"] == 1
    assert cfg["rhdb_options"]["use_hypergraph"] is True
    assert cfg["rhdb_options"]["fusion_mode"] == "gate"
    overview(cfg)
    print("Saved acmmamba_soft_cross_rhdb_architecture.png and .svg")
