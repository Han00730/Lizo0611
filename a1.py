import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
 
st.set_page_config(page_title="多股票技術分析平台", layout="wide")
st.title("📈 金融商品多股票技術分析平台")

# 股票清單
stock_list = {
    "00878": "00878.TW",
    "00912": "00912.TW",
    "0050": "0050.TW",
    "1815": "1815.TW",
    "3231": "3231.TW"
}

# 側邊欄選單
with st.sidebar:
    st.header("參數設定")
    selected_stocks = st.multiselect("選擇股票 (可複選)", list(stock_list.keys()), default=["00878"])
    start_date = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("結束日期", pd.to_datetime("2025-05-20"))
    interval_label = st.selectbox("K棒長度", ["日K", "週K", "月K"])
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo"}
    interval = interval_map[interval_label]

    st.subheader("技術指標設定")
    long_ma = st.number_input("設定計算長移動平均線(MA)的 K 棒數目 (整數, 例如 10)", min_value=1, max_value=120, value=20)
    short_ma = st.number_input("設定計算短移動平均線(MA)的 K 棒數目 (整數, 例如 2)", min_value=1, max_value=60, value=5)
    long_rsi = st.number_input("設定計算長 RSI 的 K 棒數目 (整數, 例如 10)", min_value=1, max_value=120, value=14)
    short_rsi = st.number_input("設定計算短 RSI 的 K 棒數目 (整數, 例如 2)", min_value=1, max_value=60, value=6)
    bb_period = st.number_input("布林通道週期", 1, 100, 20)
    bb_std = st.number_input("布林通道寬度 (倍數)", 0.5, 5.0, 2.0)
    macd_fast = st.number_input("MACD 快速線週期", 1, 50, 12)
    macd_slow = st.number_input("MACD 慢速線週期", 1, 50, 26)
    macd_signal = st.number_input("MACD 訊號線週期", 1, 20, 9)

    st.subheader("策略參數設定")
    stop_loss = st.number_input("停損量 (元或點數)", 0.0, 100.0, 30.0)
    trade_volume = st.number_input("購買數量 (張/口)", 1, 100, 1)

# 技術指標計算函數
@st.cache_data
def compute_indicators(df):
    df['MA_short'] = df['Close'].rolling(window=short_ma).mean()
    df['MA_long'] = df['Close'].rolling(window=long_ma).mean()

    delta = df['Close'].diff()
    delta_1d = delta.to_numpy().flatten()
    gain = pd.Series(np.where(delta_1d > 0, delta_1d, 0), index=delta.index)
    loss = pd.Series(np.where(delta_1d < 0, -delta_1d, 0), index=delta.index)

    avg_gain_short = gain.rolling(window=short_rsi).mean()
    avg_loss_short = loss.rolling(window=short_rsi).mean()
    rs_short = avg_gain_short / avg_loss_short
    df['RSI_short'] = 100 - (100 / (1 + rs_short))

    avg_gain_long = gain.rolling(window=long_rsi).mean()
    avg_loss_long = loss.rolling(window=long_rsi).mean()
    rs_long = avg_gain_long / avg_loss_long
    df['RSI_long'] = 100 - (100 / (1 + rs_long))

    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    std = df['Close'].rolling(window=bb_period).std()
    if isinstance(std, pd.DataFrame):
        std = std.squeeze()
    df['BB_Upper'] = df['BB_Middle'] + bb_std * std
    df['BB_Lower'] = df['BB_Middle'] - bb_std * std

    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['Return'] = df['Close'].pct_change()
    df['Cumulative Return'] = (1 + df['Return']).cumprod() - 1

    return df

# 主畫面內容
for stock_key in selected_stocks:
    st.markdown(f"## {stock_key} 股票分析")
    stock_code = stock_list[stock_key]
    df = yf.download(stock_code, start=start_date, end=end_date, interval=interval)

    if df.empty:
        st.warning(f"{stock_key} 無資料")
        continue

    df = compute_indicators(df)

    tabs = st.tabs(["📊 K線與MA", "📈 RSI與布林通道", "💹 MACD與策略", "📉 累積報酬與成交量"])

    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_short'], line=dict(color='orange'), name=f'MA{short_ma}'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_long'], line=dict(color='blue'), name=f'MA{long_ma}'))
        fig.update_layout(title=f"{stock_key} K 線圖與 MA", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_short'], name='RSI短期'))
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_long'], name='RSI長期'))

        buy_signal = df[df['RSI_short'] < 30]
        sell_signal = df[df['RSI_short'] > 70]

        rsi_fig.add_trace(go.Scatter(x=buy_signal.index, y=buy_signal['RSI_short'], mode='markers', marker=dict(color='blue', size=8), name='買點 (RSI<30)'))
        rsi_fig.add_trace(go.Scatter(x=sell_signal.index, y=sell_signal['RSI_short'], mode='markers', marker=dict(color='red', size=8), name='賣點 (RSI>70)'))
        rsi_fig.update_layout(title="RSI 指標", height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)

        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價'))
        bb_fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB上軌'))
        bb_fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB中軌'))
        bb_fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB下軌'))
        bb_fig.update_layout(title="布林通道", height=300)
        st.plotly_chart(bb_fig, use_container_width=True)

    with tabs[2]:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal'))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram'))
        fig_macd.update_layout(title="MACD 指標圖", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        st.markdown("### 策略參數")
        st.write(f"- 停損量：{stop_loss} 元/點")
        st.write(f"- 購買數量：{trade_volume} 張/口")

    with tabs[3]:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative Return'], mode='lines', name='累積報酬率'))
        fig_cum.update_layout(title="累積報酬率", height=300)
        st.plotly_chart(fig_cum, use_container_width=True)

        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量'))
        volume_fig.update_layout(title="成交量圖", height=200)
        st.plotly_chart(volume_fig, use_container_width=True)
