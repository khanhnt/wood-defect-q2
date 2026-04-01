"""Basic checks for hybrid ablation config coverage."""

from pathlib import Path

from src.utils.config import load_yaml


HYBRID_CONFIGS = [
    "configs/train_hybrid_cnn_micro.yaml",
    "configs/train_hybrid_cnn_transformer_micro.yaml",
    "configs/train_hybrid_cnn_p2_micro.yaml",
    "configs/train_hybrid_cnn_transformer_p2_micro.yaml",
    "configs/train_hybrid_cnn_smoke.yaml",
    "configs/train_hybrid_cnn_transformer_smoke.yaml",
    "configs/train_hybrid_cnn_p2_smoke.yaml",
    "configs/train_hybrid_cnn_transformer_p2_smoke.yaml",
    "configs/train_hybrid_cnn.yaml",
    "configs/train_hybrid_cnn_transformer.yaml",
    "configs/train_hybrid_cnn_p2.yaml",
    "configs/train_hybrid_cnn_transformer_p2.yaml",
    "configs/eval_hybrid_cnn.yaml",
    "configs/eval_hybrid_cnn_transformer.yaml",
    "configs/eval_hybrid_cnn_p2.yaml",
    "configs/eval_hybrid_cnn_transformer_p2.yaml",
    "configs/eval_hybrid_cnn_vnwoodknot.yaml",
    "configs/eval_hybrid_cnn_transformer_vnwoodknot.yaml",
    "configs/eval_hybrid_cnn_p2_vnwoodknot.yaml",
    "configs/eval_hybrid_cnn_transformer_p2_vnwoodknot.yaml",
]


def test_hybrid_ablation_config_files_exist_and_load() -> None:
    for path in HYBRID_CONFIGS:
        assert Path(path).exists(), path
        config = load_yaml(path)
        assert config["model"]["name"] == "hybrid_detector"
        assert "num_classes" in config["model"]
