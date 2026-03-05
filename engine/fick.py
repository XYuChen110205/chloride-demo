"""
Fick 第二定律解析解与模拟数据生成（mock data，仅用于开发与测试）。
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import erfc

logger = logging.getLogger("engine.fick")


def fick_analytical(
    x_mm: float,
    t_days: float,
    D0: float,
    Cs: float,
    n: float = 0.3,
    t0_days: float = 28.0,
) -> float:
    """单点 Fick 解析解，返回浓度值（占混凝土质量，%）。

    时变扩散系数: D(t) = D0 * (t0/t)^n。
    C(x,t) = Cs * erfc(x / (2*sqrt(D(t)*t)))，C0=0。

    Args:
        x_mm: 渗透深度，mm。
        t_days: 暴露时间，天。
        D0: 参考扩散系数，m²/s。
        Cs: 表面氯离子浓度，%。
        n: 时间衰减指数。
        t0_days: 参考龄期，天。

    Returns:
        该深度、时间下的氯离子浓度，%。
    """
    if t_days <= 0:
        return 0.0
    x_m = x_mm / 1000.0
    t_sec = t_days * 86400.0
    t0_sec = t0_days * 86400.0
    D_t = D0 * (t0_sec / t_sec) ** n
    return float(Cs * erfc(x_m / (2.0 * np.sqrt(D_t * t_sec))))


def generate_mock_data(
    n_samples: int = 8,
    time_days_list: list | None = None,
    depths_mm_list: list | None = None,
    noise_std: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟氯离子浓度数据（mock data）。

    列: sample_id, w_c_ratio, D0, Cs_surface, time_days, depth_mm, concentration。

    Args:
        n_samples: 工况数。
        time_days_list: 暴露时间序列（天）；默认 60～1260，步长 60。
        depths_mm_list: 渗透深度序列（mm）；默认 1～30，步长 1。
        noise_std: 浓度噪声标准差；为 0 时使用固定 n 以保证与解析解一致。
        seed: 随机种子。

    Returns:
        模拟数据 DataFrame。
    """
    rng = np.random.default_rng(seed)

    if time_days_list is None:
        time_days_list = list(range(60, 1261, 60))
    if depths_mm_list is None:
        depths_mm_list = [float(x) for x in range(1, 31)]

    wc_ratios = np.linspace(0.30, 0.55, n_samples)
    D0_values = np.linspace(0.8e-12, 6.0e-12, n_samples)
    Cs_values = np.linspace(0.4, 1.8, n_samples)

    records: list[dict] = []
    t0_days = 28.0

    for sid in range(n_samples):
        wc = round(float(wc_ratios[sid]), 3)
        D0 = float(D0_values[sid])
        Cs = float(Cs_values[sid])
        if noise_std == 0:
            n = 0.4
        else:
            n = 0.3 + 0.2 * float(rng.random())

        for t_day in time_days_list:
            for x_mm in depths_mm_list:
                C = fick_analytical(
                    x_mm=float(x_mm),
                    t_days=float(t_day),
                    D0=D0,
                    Cs=Cs,
                    n=n,
                    t0_days=t0_days,
                )
                if noise_std > 0:
                    C = C + float(rng.normal(0, noise_std))
                C = max(0.0, C)

                records.append({
                    "sample_id": sid,
                    "w_c_ratio": wc,
                    "D0": D0,
                    "Cs_surface": Cs,
                    "time_days": int(t_day),
                    "depth_mm": int(x_mm) if x_mm == int(x_mm) else x_mm,
                    "concentration": round(C, 5),
                })

    df = pd.DataFrame(records)
    logger.info("generate_mock_data: n_samples=%s, rows=%s", n_samples, len(df))
    return df
