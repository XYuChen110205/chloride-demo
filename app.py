"""
氯离子耐久性评估系统 - 演示版
Streamlit 独立演示：数据生成、模型训练、浓度预测、寿命评估
"""

import sys
from pathlib import Path

# 确保 engine 在路径中（streamlit_demo 根目录为工作目录）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.fick import generate_mock_data
from engine.predictor import ChloridePredictor, PredictInput
from engine.trainer import TrainConfig, train_in_memory

# 页面配置
st.set_page_config(
    page_title="氯离子耐久性评估系统 - 演示版",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化 session_state
if "demo_df" not in st.session_state:
    st.session_state.demo_df = None
if "trained_state" not in st.session_state:
    st.session_state.trained_state = None  # (state_dict, scaler, meta, metrics)

SEQ_LEN = 5  # 与 engine 默认一致，预测时 history 需至少 SEQ_LEN 条


def _build_history_from_df(df: pd.DataFrame, sample_id: int, depths_mm: list) -> list:
    """从 DataFrame 中抽取指定 sample_id 和深度的历史序列，供预测使用。"""
    history = []
    for depth in depths_mm:
        sub = df[(df["sample_id"] == sample_id) & (df["depth_mm"] == depth)].sort_values("time_days")
        for _, row in sub.iterrows():
            history.append({
                "time_days": int(row["time_days"]),
                "depth_mm": float(row["depth_mm"]),
                "w_c_ratio": float(row["w_c_ratio"]),
                "concentration": float(row["concentration"]),
            })
    return history


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("氯离子耐久性评估系统")
    st.caption("演示版 · 无数据库 / 无持久存储")
    st.divider()
    st.subheader("项目说明")
    st.markdown("""
    本演示基于 **Fick 第二定律** 解析解生成模拟氯离子渗透数据，
    使用 **LSTM / Attention-LSTM / GRU / BP** 等模型进行时序预测与寿命评估。

    **使用流程：**
    1. **数据生成**：调整样本数、噪声，生成模拟数据  
    2. **模型训练**：选择模型与超参，在页面上训练（结果仅存于当前会话）  
    3. **浓度预测**：基于训练好的模型做多步预测，查看浓度-时间/深度曲线  
    4. **寿命评估**：输入保护层厚度与阈值，得到风险等级与预估年限  
    """)
    st.divider()
    st.subheader("全局参数")
    st.markdown("- **序列长度 seq_len**：5（输入时间步）")
    st.markdown("- **预测步长 pred_len**：1")
    st.markdown("- 训练好的模型保存在 **内存**（session_state），刷新页面会丢失。")
    st.divider()
    if st.session_state.trained_state is not None:
        _, _, meta, metrics = st.session_state.trained_state
        st.success(f"已加载模型：{meta.get('model_type', '?')}")
        st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    else:
        st.info("尚未训练模型，请先在「模型训练」页训练。")

# ---------- 主标题 ----------
st.title("氯离子耐久性评估系统 - 演示版")
st.caption("数据生成 → 模型训练 → 浓度预测 → 寿命评估")

tab1, tab2, tab3, tab4 = st.tabs(["数据生成", "模型训练", "浓度预测", "寿命评估"])

# ========== Tab1: 数据生成 ==========
with tab1:
    st.subheader("模拟数据生成")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_samples = st.slider("样本数（工况数）", min_value=2, max_value=20, value=8, step=1)
        noise_std = st.slider("浓度噪声标准差", min_value=0.0, max_value=0.05, value=0.015, step=0.005)
        seed = st.number_input("随机种子", min_value=0, value=42, step=1)
        if st.button("一键生成模拟数据", type="primary"):
            with st.spinner("生成中…"):
                df = generate_mock_data(n_samples=n_samples, noise_std=noise_std, seed=seed)
                st.session_state.demo_df = df
                st.success(f"已生成 {len(df)} 条记录，{n_samples} 个样本。")
                st.rerun()
    with col2:
        if st.session_state.demo_df is not None:
            df = st.session_state.demo_df
            st.dataframe(df.head(200), use_container_width=True, height=300)
            st.caption("（仅展示前 200 行）")
            st.subheader("基础统计")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("请点击「一键生成模拟数据」生成数据后再查看表格与统计。")

# ========== Tab2: 模型训练 ==========
with tab2:
    st.subheader("模型训练")
    if st.session_state.demo_df is None:
        st.warning("请先在「数据生成」页生成数据。")
        # 仍允许用默认 mock 数据训练
        use_mock = st.checkbox("使用内置 mock 数据训练（不依赖已生成数据）", value=True)
        df_for_train = None
        if use_mock:
            df_for_train = generate_mock_data(n_samples=8, noise_std=0.015, seed=42)
    else:
        df_for_train = st.session_state.demo_df
        use_mock = False

    if df_for_train is not None or st.session_state.demo_df is not None:
        actual_df = df_for_train if df_for_train is not None else st.session_state.demo_df
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "模型类型",
                options=["attention_lstm", "simple_lstm", "gru", "bp"],
                format_func=lambda x: {"attention_lstm": "Attention-LSTM", "simple_lstm": "LSTM", "gru": "GRU", "bp": "BP"}[x],
            )
            epochs = st.slider("训练轮数 epochs", min_value=10, max_value=500, value=100, step=10)
            hidden_size = st.slider("隐藏层大小 hidden_size", min_value=32, max_value=256, value=128, step=32)
            lr = st.select_slider("学习率", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, format_func=lambda x: str(x))
        with col2:
            pass  # 预留

        if st.button("开始训练", type="primary"):
            config = TrainConfig(
                data_path="mock",  # 下面传 df，不读文件
                model_type=model_type,
                seq_len=SEQ_LEN,
                pred_len=1,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.1,
                epochs=epochs,
                learning_rate=lr,
                batch_size=32,
                test_ratio=0.2,
            )
            progress_bar = st.progress(0.0, text="训练中…")
            loss_placeholder = st.empty()
            loss_history = []

            def progress_cb(epoch: int, total: int, loss: float):
                loss_history.append(loss)
                progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}，Loss: {loss:.6f}")
                if epoch % 10 == 0 or epoch == total:
                    loss_placeholder.line_chart(loss_history)

            try:
                state_dict, scaler, meta, metrics, loss_history, elapsed = train_in_memory(
                    config, df=actual_df, progress_callback=progress_cb
                )
                st.session_state.trained_state = (state_dict, scaler, meta, metrics)
                progress_bar.progress(1.0, text="训练完成")
                loss_placeholder.empty()
                st.success(f"训练完成，耗时 {elapsed:.1f} 秒。")
                # Loss 曲线
                fig_loss = go.Figure(data=[go.Scatter(y=loss_history, mode="lines", name="Loss")])
                fig_loss.update_layout(
                    title="训练 Loss 曲线",
                    xaxis_title="Epoch",
                    yaxis_title="MSE Loss",
                    template="plotly_white",
                    height=300,
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                # 指标
                st.subheader("评估指标")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{metrics['mae']:.4f}")
                m2.metric("RMSE", f"{metrics['rmse']:.4f}")
                m3.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                m4.metric("R²", f"{metrics['r2']:.4f}")
            except Exception as e:
                st.error(f"训练失败：{e}")
                import traceback
                st.code(traceback.format_exc())

# ========== Tab3: 浓度预测 ==========
with tab3:
    st.subheader("浓度预测")
    if st.session_state.trained_state is None:
        st.warning("请先在「模型训练」页完成训练。")
    else:
        state_dict, scaler, meta, _ = st.session_state.trained_state
        predictor = ChloridePredictor.from_memory(state_dict, scaler, meta)
        df_src = st.session_state.demo_df
        if df_src is None:
            df_src = generate_mock_data(n_samples=8, noise_std=0.015, seed=42)
        sample_id = st.number_input("选用样本 ID（用于历史序列）", min_value=0, max_value=max(0, int(df_src["sample_id"].max())), value=0, step=1)
        predict_steps = st.number_input("预测步数", min_value=1, max_value=100, value=20, step=1)
        time_interval_days = st.number_input("时间间隔（天）", min_value=1, value=120, step=1)
        depths_str = st.text_input("深度列表（mm），逗号分隔", value="5, 10, 15, 20, 25")
        try:
            depths_mm = [float(x.strip()) for x in depths_str.split(",") if x.strip()]
        except ValueError:
            depths_mm = [5.0, 10.0, 15.0]
        # 只保留数据中存在的深度，且每个深度至少 SEQ_LEN 个时间点
        available = df_src[(df_src["sample_id"] == sample_id)]["depth_mm"].unique()
        depths_mm = [d for d in depths_mm if d in available]
        if not depths_mm:
            depths_mm = [float(available[0])] if len(available) > 0 else [5.0]
        history = _build_history_from_df(df_src, sample_id, depths_mm)
        if len(history) < SEQ_LEN * len(depths_mm):
            st.warning("当前样本+深度组合的历史点数不足，已自动选用可用深度。")
        if st.button("执行预测", type="primary"):
            inp = PredictInput(history=history, predict_steps=predict_steps, time_interval_days=time_interval_days, depths_mm=depths_mm)
            out = predictor.predict(inp)
            preds = out.predictions
            if not preds:
                st.error("没有产生预测结果，请检查历史数据是否满足序列长度要求。")
            else:
                pdf = pd.DataFrame(preds)
                # 浓度-时间曲线（按深度分线）
                fig_t = go.Figure()
                for depth in pdf["depth_mm"].unique():
                    sub = pdf[pdf["depth_mm"] == depth]
                    fig_t.add_trace(go.Scatter(x=sub["time_days"], y=sub["concentration"], mode="lines+markers", name=f"深度 {depth} mm"))
                fig_t.update_layout(
                    title="浓度 - 时间曲线",
                    xaxis_title="时间 (天)",
                    yaxis_title="浓度 (%)",
                    template="plotly_white",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_t, use_container_width=True)
                # 浓度-深度曲线（按时间分线，取若干时间点）
                times = sorted(pdf["time_days"].unique())
                step = max(1, len(times) // 5)
                times_show = times[::step][:5]
                fig_d = go.Figure()
                for t in times_show:
                    sub = pdf[pdf["time_days"] == t]
                    sub = sub.sort_values("depth_mm")
                    fig_d.add_trace(go.Scatter(x=sub["depth_mm"], y=sub["concentration"], mode="lines+markers", name=f"t={t} 天"))
                fig_d.update_layout(
                    title="浓度 - 深度曲线（选定时刻）",
                    xaxis_title="深度 (mm)",
                    yaxis_title="浓度 (%)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_d, use_container_width=True)
                st.dataframe(pdf, use_container_width=True)

# ========== Tab4: 寿命评估 ==========
with tab4:
    st.subheader("寿命评估")
    if st.session_state.trained_state is None:
        st.warning("请先在「模型训练」页完成训练。")
    else:
        state_dict, scaler, meta, _ = st.session_state.trained_state
        predictor = ChloridePredictor.from_memory(state_dict, scaler, meta)
        df_src = st.session_state.demo_df
        if df_src is None:
            df_src = generate_mock_data(n_samples=8, noise_std=0.015, seed=42)
        sample_id = st.number_input("选用样本 ID（寿命）", min_value=0, max_value=max(0, int(df_src["sample_id"].max())), value=0, step=1, key="life_sid")
        cover_depth_mm = st.number_input("保护层厚度 (mm)", min_value=1.0, max_value=80.0, value=30.0, step=1.0)
        threshold = st.number_input("浓度阈值 (%)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
        design_life = st.number_input("设计寿命 (年)", min_value=10, value=50, step=5)
        depths_for_history = [cover_depth_mm]
        # 若数据中无该深度，取最接近的深度
        available_depths = sorted(df_src[(df_src["sample_id"] == sample_id)]["depth_mm"].unique())
        depth_eval = cover_depth_mm
        if available_depths:
            depth_eval = min(available_depths, key=lambda d: abs(d - cover_depth_mm))
        depths_for_history = [depth_eval]
        history_life = _build_history_from_df(df_src, sample_id, depths_for_history)
        if depth_eval != cover_depth_mm:
            st.caption(f"数据中无 {cover_depth_mm} mm 深度，将使用最接近的 {depth_eval} mm 进行评估。")
        if st.button("执行寿命评估", type="primary"):
            inp = PredictInput(history=history_life, predict_steps=2000, time_interval_days=365, depths_mm=depths_for_history)
            life = predictor.estimate_corrosion_time(
                inp, cover_depth_mm=depth_eval, threshold=threshold, design_life_years=float(design_life)
            )
            risk_cn = {"safe": "安全", "warning": "注意", "danger": "危险"}
            st.metric("风险等级", risk_cn.get(life.risk_level, life.risk_level))
            st.metric("预估达到阈值年限（年）", life.estimated_years)
            st.metric("设计寿命（年）", design_life)
            # 浓度-时间线图（含阈值线）
            tl = life.concentration_timeline
            if tl:
                df_tl = pd.DataFrame(tl)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_tl["time_days"], y=df_tl["concentration"], mode="lines+markers", name="预测浓度"))
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"阈值 {threshold}%")
                fig.update_layout(
                    title=f"保护层 {depth_eval} mm 处浓度时间线（阈值 {threshold}%）",
                    xaxis_title="时间 (天)",
                    yaxis_title="浓度 (%)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("时间线无数据。")
