# 氯离子耐久性评估系统 - Streamlit 演示版

独立于主项目的 Streamlit 演示应用，用于在线展示氯离子预测模型：数据生成、模型训练、浓度预测、寿命评估。

## 功能概览

- **数据生成**：滑块调节样本数、噪声与随机种子，一键生成 Fick 解析解模拟数据，展示表格与基础统计。
- **模型训练**：选择模型类型（Attention-LSTM / LSTM / GRU / BP），调节 epochs、hidden_size、学习率，在页面上实时训练，显示进度条与 Loss 曲线，完成后展示 MAE / RMSE / MAPE / R²。
- **浓度预测**：输入预测步数、时间间隔、深度列表，基于已训练模型做多步滚动预测，展示浓度-时间曲线与浓度-深度曲线（Plotly 交互图）。
- **寿命评估**：输入保护层厚度与浓度阈值，得到风险等级、预估达到阈值的年数，以及浓度时间线图（含阈值线）。

## 本地运行

```bash
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

## 部署到 Streamlit Cloud

1. 将本目录（或包含 `streamlit_demo` 的仓库）推送到 GitHub。
2. 登录 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号授权。
3. 点击 “New app”，选择仓库与分支，**Root directory** 填 `streamlit_demo`（若应用在仓库根目录下的 `streamlit_demo` 中）。
4. **Main file path** 填 `app.py`。
5. 若需指定 Python 版本，在仓库根目录添加 `.streamlit/config.toml` 或于 Cloud 设置中指定 Python 3.10+。
6. 部署后即可通过分享链接访问。

## 依赖说明

- **不依赖数据库**：所有数据在会话内生成或保存在内存。
- **不依赖 FastAPI**：仅 Streamlit 单应用。
- **模型与 scaler**：训练结果保存在 `st.session_state`，不写磁盘，适合 Streamlit Cloud 无持久存储环境；刷新页面会丢失，需重新训练。

## 目录结构

```
streamlit_demo/
├── app.py              # 主程序（4 个 tab）
├── requirements.txt    # 依赖
├── README.md           # 本说明
└── engine/             # 复制的 engine 层（可独立运行）
    ├── __init__.py
    ├── fick.py         # Fick 解析解与 mock 数据生成
    ├── models.py       # LSTM / GRU / BP 等模型
    ├── dataset.py      # 时序数据集与 DataLoader
    ├── trainer.py      # 训练流程（含 train_in_memory）
    └── predictor.py    # 预测与寿命评估（含 from_memory）
```

## 界面说明

- 页面标题：**氯离子耐久性评估系统 - 演示版**，中文界面。
- 侧边栏：全局参数说明、项目介绍、当前是否已加载模型及 R²。
- 图表均使用 Plotly，支持缩放、悬停等交互。
