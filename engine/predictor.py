"""
预测封装：加载已训练模型与 scaler，多步滚动预测与寿命评估。
支持从内存加载（state_dict + scaler + meta），用于 Streamlit 演示。
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engine.models import MODEL_REGISTRY

logger = logging.getLogger("engine.predictor")

FEATURE_COLS = ["time_days", "depth_mm", "w_c_ratio", "concentration"]
CONC_IDX = 3


@dataclass
class PredictInput:
    """预测输入。"""

    history: list  # time_days, depth_mm, concentration, w_c_ratio
    predict_steps: int
    time_interval_days: int
    depths_mm: list


@dataclass
class PredictOutput:
    """预测输出。"""

    predictions: list
    model_type: str
    confidence: float | None = None


@dataclass
class LifeEstimation:
    """寿命评估结果。"""

    estimated_days: int
    estimated_years: float
    risk_level: str  # "safe" | "warning" | "danger"
    threshold_used: float
    cover_depth_mm: float
    concentration_timeline: list


def _build_model_from_meta(meta: dict[str, Any]) -> torch.nn.Module:
    """根据 meta 构建模型结构。"""
    model_type = meta["model_type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type in meta: {model_type}")
    cls = MODEL_REGISTRY[model_type]
    if model_type == "bp":
        return cls(
            input_size=meta["input_size"],
            seq_len=meta["seq_len"],
            pred_len=meta["pred_len"],
        )
    return cls(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        pred_len=meta["pred_len"],
        dropout=0.1,
    )


def _inverse_scale_concentration(scaler: Any, scaled: float) -> float:
    """将归一化后的浓度反变换为原始量纲。"""
    lo = scaler.data_min_[CONC_IDX]
    hi = scaler.data_max_[CONC_IDX]
    return float(scaled * (hi - lo) + lo)


class ChloridePredictor:
    """加载已训练模型与 scaler，提供 predict 与 estimate_corrosion_time。"""

    @classmethod
    def from_memory(
        cls,
        state_dict: dict,
        scaler: Any,
        meta: dict[str, Any],
    ) -> "ChloridePredictor":
        """从内存中的 state_dict、scaler、meta 构建预测器（不读文件）。"""
        self = cls.__new__(cls)
        self.scaler = scaler
        self.meta = meta
        self.seq_len = meta["seq_len"]
        self.pred_len = meta["pred_len"]
        self.model = _build_model_from_meta(meta)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_type = meta["model_type"]
        logger.info("ChloridePredictor from_memory: model_type=%s", self.model_type)
        return self

    def __init__(self, model_path: str, scaler_path: str) -> None:
        """从 model_path / scaler_path 加载；model_meta.pkl 需与 model.pth 同目录。"""
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"model_path not found: {model_path}")
        scaler_path = Path(scaler_path)
        if not scaler_path.is_file():
            raise FileNotFoundError(f"scaler_path not found: {scaler_path}")
        meta_path = model_path.parent / "model_meta.pkl"
        if not meta_path.is_file():
            raise FileNotFoundError(f"model_meta.pkl not found next to model: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.meta = meta
        self.seq_len = meta["seq_len"]
        self.pred_len = meta["pred_len"]
        self.model = _build_model_from_meta(meta)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except TypeError:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_type = meta["model_type"]
        logger.info("ChloridePredictor loaded: model_type=%s", self.model_type)

    def predict(self, input_data: PredictInput) -> PredictOutput:
        """多步滚动预测；history 按 depth_mm 分组，每个深度独立滚动。"""
        predictions: list = []
        by_depth: dict = {}
        for row in input_data.history:
            d = float(row["depth_mm"])
            if d not in by_depth:
                by_depth[d] = []
            by_depth[d].append(dict(row))
        for depth in input_data.depths_mm:
            depth = float(depth)
            if depth not in by_depth or len(by_depth[depth]) < self.seq_len:
                continue
            seq = sorted(by_depth[depth], key=lambda x: x["time_days"])
            w_c = seq[-1].get("w_c_ratio", 0.45)
            max_t = max(r["time_days"] for r in seq)
            window = seq[-self.seq_len :]
            for step in range(input_data.predict_steps):
                t = max_t + (step + 1) * input_data.time_interval_days
                arr = np.array(
                    [[r["time_days"], r["depth_mm"], r["w_c_ratio"], r["concentration"]] for r in window],
                    dtype=np.float32,
                )
                arr_scaled = self.scaler.transform(arr)
                x = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(x).cpu().numpy().flatten()[0]
                c = _inverse_scale_concentration(self.scaler, float(out))
                c = max(0.0, c)
                predictions.append({"time_days": int(t), "depth_mm": depth, "concentration": round(c, 6)})
                window = window[1:] + [{"time_days": t, "depth_mm": depth, "w_c_ratio": w_c, "concentration": c}]
        return PredictOutput(predictions=predictions, model_type=self.model_type, confidence=None)

    def estimate_corrosion_time(
        self,
        input_data: PredictInput,
        cover_depth_mm: float,
        threshold: float = 0.6,
        max_years: int = 100,
        design_life_years: float = 50.0,
    ) -> LifeEstimation:
        """在保护层深度处滚动预测，直到浓度达到阈值；并计算风险等级。"""
        max_days = max_years * 365
        interval = max(1, input_data.time_interval_days)
        max_steps = min(5000, max(1, max_days // interval))
        input_single = PredictInput(
            history=input_data.history,
            predict_steps=max_steps,
            time_interval_days=interval,
            depths_mm=[cover_depth_mm],
        )
        out = self.predict(input_single)
        timeline: list = []
        estimated_days = max_days
        for p in out.predictions:
            timeline.append({"time_days": p["time_days"], "concentration": p["concentration"]})
            if p["concentration"] >= threshold:
                estimated_days = p["time_days"]
                break
        estimated_years = estimated_days / 365.0
        if estimated_years > design_life_years * 1.2:
            risk_level = "safe"
        elif estimated_years > design_life_years * 0.8:
            risk_level = "warning"
        else:
            risk_level = "danger"
        return LifeEstimation(
            estimated_days=int(estimated_days),
            estimated_years=round(estimated_years, 2),
            risk_level=risk_level,
            threshold_used=threshold,
            cover_depth_mm=cover_depth_mm,
            concentration_timeline=timeline,
        )


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent
    model_path = root / "saved_models" / "model.pth"
    scaler_path = root / "saved_models" / "scaler.pkl"
    if not model_path.is_file():
        print("Run python -m engine.trainer first to create saved_models/")
        raise SystemExit(1)
    p = ChloridePredictor(str(model_path), str(scaler_path))
    inp = PredictInput(
        history=[
            {"time_days": 60 + i * 60, "depth_mm": 5.0, "w_c_ratio": 0.45, "concentration": 0.2 + 0.02 * i}
            for i in range(5)
        ],
        predict_steps=10,
        time_interval_days=120,
        depths_mm=[5.0, 10.0],
    )
    out = p.predict(inp)
    print("Predictions count:", len(out.predictions))
    print("Sample:", out.predictions[:3])
    life = p.estimate_corrosion_time(inp, cover_depth_mm=30.0, threshold=0.6)
    print("Life estimation:", life.estimated_years, "years, risk:", life.risk_level)


if __name__ == "__main__":
    main()
