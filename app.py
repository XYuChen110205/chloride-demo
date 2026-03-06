"""
Streamlit 快速预览：8 个功能页面，对齐 React 前端能力。
完整功能请使用 React 前端系统。需先启动后端：uvicorn backend.app.main:app --reload
"""
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"
ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="氯离子预测系统", layout="wide", page_icon="🧪")

st.info("本 Streamlit 为快速预览界面，完整功能请使用 React 前端系统。")

sidebar = st.sidebar
sidebar.title("功能导航")
PAGES = [
    "🏠 首页概览",
    "📁 数据管理",
    "🔧 模型训练",
    "📊 实验对比",
    "🔬 UCI 对照实验",
    "⏱ 寿命预测",
    "📈 预测分析",
    "ℹ️ 关于系统",
]
page = sidebar.radio("选择页面", PAGES, label_visibility="collapsed")


def fetch_json(method: str, url: str, **kwargs) -> dict | None:
    try:
        r = requests.request(method, url, timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


# ---------- 1. 首页概览 ----------
if page == "🏠 首页概览":
    st.title("🏠 首页概览")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型数量", "9")
    c2.metric("最佳 R² (Random)", "0.93")
    c3.metric("样本量", "918")
    c4.metric("CI 覆盖率", "93.6%")
    st.subheader("系统简介")
    st.write(
        "基于知识-数据双驱动的混凝土氯离子侵蚀预测与服役寿命评估系统。"
        "融合 Fick 物理模型与 XGBoost、GPR、BP、LSTM/GRU/AttentionLSTM、PatchTST 等 9 类模型，"
        "支持半合成预训练与物理约束微调，提供 GroupKFold 严格泛化评估、MC-Dropout 不确定性量化与 SHAP 可解释性。"
        "工程系统包含数据管理、模型训练、版本回滚、寿命预测与 React 前端展示。"
    )

# ---------- 2. 数据管理 ----------
elif page == "📁 数据管理":
    st.title("📁 数据管理")
    st.divider()
    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            st.subheader("前 10 行预览")
            st.dataframe(df.head(10), use_container_width=True)
            st.subheader("统计信息")
            st.write(df.describe())
            if st.button("确认导入"):
                st.success("数据已导入系统（演示：未实际写入后端）")
        except Exception as e:
            st.error(f"解析 CSV 失败: {e}")

# ---------- 3. 模型训练 ----------
elif page == "🔧 模型训练":
    st.title("🔧 模型训练")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("一键全量训练", use_container_width=True):
            with st.spinner("训练中..."):
                r = fetch_json("POST", f"{API_BASE}/train/full")
            if r and r.get("status") in ("started", "ok"):
                st.success("全量训练已启动（后台运行，可稍后在实验对比页查看结果）")
            elif r and r.get("status") == "busy":
                st.warning(r.get("message", "已有训练任务进行中"))
    with col2:
        if st.button("一键 UCI 实验", use_container_width=True):
            with st.spinner("训练中..."):
                r = fetch_json("POST", f"{API_BASE}/train/uci")
            if r and r.get("status") in ("started", "ok"):
                st.success("UCI 实验已启动（后台运行）")
            elif r and r.get("status") == "busy":
                st.warning(r.get("message", "已有训练任务进行中"))
    st.subheader("训练历史")
    hist = fetch_json("GET", f"{API_BASE}/train/history")
    if hist and hist.get("items"):
        rows = [
            {"训练时间": str(h.get("timestamp", "")), "类型": h.get("trigger_type", ""), "最佳 R²": h.get("best_r2")}
            for h in hist["items"]
        ]
        st.table(pd.DataFrame(rows).iloc[::-1])
    else:
        st.caption("暂无训练历史，完成一次全量训练后将自动记录。")

# ---------- 4. 实验对比 ----------
elif page == "📊 实验对比":
    st.title("📊 实验对比")
    st.divider()
    st.header("氯离子数据集 9 模型")
    data = fetch_json("GET", f"{API_BASE}/experiment/results-summary")
    if data:
        fc = data.get("full_comparison") or {}
        if not fc:
            st.warning("暂无 full_comparison 结果，请先运行全量训练。")
        else:
            rows = []
            for model, metrics in fc.items():
                rows.append({
                    "模型": model,
                    "R²_mean": round(metrics.get("R2_mean"), 2) if metrics.get("R2_mean") is not None else None,
                    "R²_std": round(metrics.get("R2_std"), 2) if metrics.get("R2_std") is not None else None,
                    "RMSE_mean": round(metrics.get("RMSE_mean"), 2) if metrics.get("RMSE_mean") is not None else None,
                    "MAE_mean": round(metrics.get("MAE_mean"), 2) if metrics.get("MAE_mean") is not None else None,
                })
            st.dataframe(rows, use_container_width=True)
            if rows:
                df = pd.DataFrame(rows)
                st.bar_chart(df.set_index("模型")["R²_mean"])

# ---------- 5. UCI 对照实验 ----------
elif page == "🔬 UCI 对照实验":
    st.title("🔬 UCI 对照实验")
    st.divider()
    st.header("6 模型 R² 对比")
    data = fetch_json("GET", f"{API_BASE}/experiment/uci-results")
    if data:
        rows = []
        for model, metrics in data.items():
            rows.append({
                "模型": model,
                "R²_mean": round(metrics.get("R2_mean"), 2) if metrics.get("R2_mean") is not None else None,
                "RMSE_mean": round(metrics.get("RMSE_mean"), 2) if metrics.get("RMSE_mean") is not None else None,
            })
        st.dataframe(rows, use_container_width=True)
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.set_index("模型")["R²_mean"])
    else:
        st.warning("暂无 UCI 结果，请先运行「一键 UCI 实验」。")

# ---------- 6. 寿命预测 ----------
elif page == "⏱ 寿命预测":
    st.title("⏱ 寿命预测")
    st.divider()

    MODEL_OPTIONS = [
        "both", "fick", "xgboost", "gpr", "bp",
        "simple_lstm", "gru", "attention_lstm", "attention_lstm_ft", "patchtst",
    ]
    MODEL_LABELS = [
        "双模型协同 (both)", "Fick 扩散", "XGBoost", "GPR", "BP",
        "SimpleLSTM", "GRU", "AttentionLSTM", "AttentionLSTM 微调", "PatchTST",
    ]
    choice_map = dict(zip(MODEL_LABELS, MODEL_OPTIONS))

    col_left, col_right = st.columns([1, 2])
    with col_left:
        with st.form("service_life_form"):
            water_cement_ratio = st.number_input("水胶比", value=0.45, min_value=0.2, max_value=1.0, step=0.01)
            depth_mm = st.number_input("保护层深度 (mm)", value=25, min_value=5, max_value=150, step=5)
            surface_chloride_g_l = st.number_input("表面氯离子浓度 (g/L)", value=5.0, min_value=0.1, max_value=50.0, step=0.5)
            temperature = st.number_input("温度 (°C)", value=20, min_value=0, max_value=50, step=1)
            cement_content = st.number_input("水泥用量 (kg/m³)", value=350, min_value=200, max_value=600, step=10)
            critical_concentration = st.number_input("临界浓度 (g/L)", value=0.6, min_value=0.1, max_value=5.0, step=0.1)
            model_label = st.selectbox("模型", MODEL_LABELS, index=0)
            model_choice = choice_map.get(model_label, "both")
            submitted = st.form_submit_button("预测")

    if submitted:
        body = {
            "water_cement_ratio": water_cement_ratio,
            "depth_mm": depth_mm,
            "surface_chloride_g_l": surface_chloride_g_l,
            "temperature": temperature,
            "cement_content": cement_content,
            "critical_concentration": critical_concentration,
            "model_choice": model_choice,
        }
        result = fetch_json("POST", f"{API_BASE}/predict/service-life", json=body)
        if result:
            if result.get("error"):
                st.error(result["error"])
            else:
                with col_left:
                    st.success(f"预估服役寿命: **{result.get('estimated_service_life_years', '—')}** 年")
                    xc = result.get("xgboost_current")
                    if xc is not None:
                        # 修复：xgboost_current 为 {"concentration": 0.xxx, "unit": "g/L"}
                        conc = xc.get("concentration") if isinstance(xc, dict) else xc
                        if conc is not None:
                            st.metric("当前浓度 (XGBoost)", f"{float(conc):.4f} g/L")
                    if result.get("critical_concentration") is not None:
                        st.metric("临界浓度", f"{result['critical_concentration']} g/L")
                    if result.get("confidence"):
                        st.caption(result["confidence"])
                with col_right:
                    ts = result.get("time_series") or {}
                    years = ts.get("years") or []
                    mean_conc = ts.get("concentration") or ts.get("mean") or []
                    mean_conc = mean_conc[:len(years)] if mean_conc else []
                    if years and mean_conc:
                        n = min(len(years), len(mean_conc))
                        years, mean_conc = years[:n], mean_conc[:n]
                        critical = float(result.get("critical_concentration", 0.6))
                        ci_lower = ts.get("ci_lower")
                        ci_upper = ts.get("ci_upper")
                        fig = go.Figure()
                        if ci_lower and ci_upper and len(ci_lower) >= n and len(ci_upper) >= n:
                            fig.add_trace(go.Scatter(
                                x=years, y=ci_upper[:n], name="95% 上界",
                                line=dict(width=0), mode="lines"
                            ))
                            fig.add_trace(go.Scatter(
                                x=years, y=ci_lower[:n], name="95% 下界",
                                line=dict(width=0), mode="lines",
                                fill="tonexty", fillcolor="rgba(56,189,248,0.2)"
                            ))
                        fig.add_trace(go.Scatter(
                            x=years, y=mean_conc, name="预测浓度",
                            line=dict(color="#0ea5e9", width=2), mode="lines+markers"
                        ))
                        fig.add_hline(
                            y=critical, line_dash="dash", line_color="#ef4444",
                            annotation_text=f"临界浓度 {critical} g/L"
                        )
                        fig.update_layout(
                            title="浓度演化曲线",
                            xaxis_title="年份",
                            yaxis_title="浓度 (g/L)",
                            hovermode="x unified",
                            height=380,
                            margin=dict(l=60, r=40, t=50, b=50),
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ---------- 7. 预测分析 ----------
elif page == "📈 预测分析":
    st.title("📈 预测分析")
    st.divider()
    fig4 = ROOT / "reports" / "figures" / "fig4_shap_xgboost.png"
    fig3 = ROOT / "reports" / "figures" / "fig3_uncertainty_band.png"
    shap_alt = ROOT / "reports" / "shap_bar_xgb.png"
    unc_alt = ROOT / "reports" / "uncertainty_plot.png"
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SHAP 特征重要性")
        if (fig4.exists() or shap_alt.exists()):
            st.image(str(fig4 if fig4.exists() else shap_alt), use_container_width=True)
        else:
            st.caption("图片未生成，请先运行 SHAP 分析脚本。路径：reports/figures/fig4_shap_xgboost.png 或 reports/shap_bar_xgb.png")
    with c2:
        st.subheader("不确定性 95% 置信区间")
        if (fig3.exists() or unc_alt.exists()):
            st.image(str(fig3 if fig3.exists() else unc_alt), use_container_width=True)
        else:
            st.caption("图片未生成，请先运行不确定性量化脚本。路径：reports/figures/fig3_uncertainty_band.png 或 reports/uncertainty_plot.png")

# ---------- 8. 关于系统 ----------
elif page == "ℹ️ 关于系统":
    st.title("ℹ️ 关于系统")
    st.divider()
    st.subheader("系统版本")
    st.write("氯离子预测与服役寿命评估系统 v1.0 | 2026")
    st.subheader("技术栈")
    st.write(
        "后端：FastAPI、PyTorch、XGBoost、scikit-learn、SHAP  |  "
        "前端：React、TypeScript、Ant Design、ECharts  |  "
        "数据：918 条真实数据 + 约 24 000 条半合成数据  |  "
        "模型：Fick、XGBoost、GPR、BP、SimpleLSTM、GRU、AttentionLSTM、PatchTST、AttentionLSTM 微调"
    )
    st.subheader("参考文献")
    st.write(
        "1. Aliasghar-Mamaghani, M. & Khalafi, M. (2026). Machine learning for chloride diffusion in concrete. *arXiv:2601.01009* (ResearchGate preprint).  \n"
        "2. Shaban, W.M., Elbaz, K., Zhou, A., & Shen, S.L. (2023). Physics-informed deep neural network for modeling the chloride diffusion in concrete. *Engineering Applications of Artificial Intelligence*, 125, 106691.  \n"
        "3. Huang, Z., Xie, S., et al. (2026). Physics-informed neural network based inversion and prediction of natural chloride diffusion in uncracked and cracked concrete systems. *Cement and Concrete Composites*. Politecnico di Milano.  \n"
        "4. Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30."
    )
    st.subheader("仓库链接")
    st.markdown("[GitHub: chloride-system](https://github.com/XYuChen110205/chloride-system)")

st.caption("完整功能请使用 React 前端系统 | © 2026 氯离子预测系统")
