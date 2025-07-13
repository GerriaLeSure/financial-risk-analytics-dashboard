import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from scipy.stats import norm

# ------------------ SETTINGS ------------------ #
st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            background-color: #0e1117;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #111;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ SAMPLE SETTINGS ------------------ #
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
SECTORS = {'AAPL': 'Tech', 'GOOGL': 'Tech', 'MSFT': 'Tech', 'TSLA': 'Auto', 'AMZN': 'Retail', 'NVDA': 'Tech'}
GEO = {'AAPL': 'USA', 'GOOGL': 'USA', 'MSFT': 'USA', 'TSLA': 'USA', 'AMZN': 'USA', 'NVDA': 'USA'}
WEIGHTS = {'AAPL': 0.2, 'GOOGL': 0.15, 'MSFT': 0.25, 'TSLA': 0.15, 'AMZN': 0.15, 'NVDA': 0.10}

# ------------------ FUNCTIONS ------------------ #
@st.cache_data(ttl=1800)
def load_data():
    data = yf.download(TICKERS, period="2y", interval="1d")['Adj Close'].dropna()
    return data

def calculate_performance_metrics(data, weights):
    returns = data.pct_change().dropna()
    weighted_returns = returns.dot(pd.Series(weights))
    cum_return = (1 + weighted_returns).cumprod()
    drawdown = cum_return / cum_return.cummax() - 1
    max_drawdown = drawdown.min()
    rolling_vol = weighted_returns.rolling(window=21).std() * np.sqrt(252)
    sharpe = weighted_returns.mean() / weighted_returns.std() * np.sqrt(252)
    sortino = weighted_returns.mean() / weighted_returns[weighted_returns < 0].std() * np.sqrt(252)
    return weighted_returns, cum_return, max_drawdown, rolling_vol, sharpe, sortino

def risk_contribution(returns, weights):
    cov_matrix = returns.cov()
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std
    contrib = weights * marginal_contrib
    return pd.Series(contrib, index=returns.columns)

# ------------------ LOAD DATA ------------------ #
data = load_data()
returns = data.pct_change().dropna()
weights_array = np.array([WEIGHTS[t] for t in TICKERS])

# ------------------ METRICS ------------------ #
port_returns, cum_perf, max_dd, rolling_vol, sharpe, sortino = calculate_performance_metrics(data[TICKERS], WEIGHTS)
risk_contrib_df = risk_contribution(returns[TICKERS], weights_array)

# ------------------ LAYOUT ------------------ #
st.title("📈 Portfolio Analysis & Optimization")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Sharpe", f"{sharpe:.2f}")
col2.metric("Sortino Ratio", f"{sortino:.2f}")
col3.metric("Max Drawdown", f"{max_dd:.2%}")
col4.metric("Volatility (21d Rolling)", f"{rolling_vol.iloc[-1]:.2%}")

# ------------------ PERFORMANCE CHART ------------------ #
perf_fig = go.Figure()
perf_fig.add_trace(go.Scatter(x=cum_perf.index, y=cum_perf, name="Cumulative Return"))
perf_fig.update_layout(template="plotly_dark", title="Portfolio Cumulative Performance", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(perf_fig, use_container_width=True)

# ------------------ ALLOCATION PIE ------------------ #
st.subheader("Asset Allocation")
alloc_fig = go.Figure(data=[
    go.Pie(labels=list(WEIGHTS.keys()), values=list(WEIGHTS.values()), hole=0.5, textinfo="label+percent")
])
alloc_fig.update_layout(template="plotly_dark")
st.plotly_chart(alloc_fig, use_container_width=True)

# ------------------ SECTOR ALLOCATION ------------------ #
st.subheader("Sector Exposure")
sector_weights = {}
for ticker, weight in WEIGHTS.items():
    sector = SECTORS[ticker]
    sector_weights[sector] = sector_weights.get(sector, 0) + weight

sector_fig = go.Figure([go.Bar(x=list(sector_weights.keys()), y=list(sector_weights.values()))])
sector_fig.update_layout(template="plotly_dark", xaxis_title="Sector", yaxis_title="Weight")
st.plotly_chart(sector_fig, use_container_width=True)

# ------------------ RISK CONTRIBUTION ------------------ #
st.subheader("Risk Contribution by Asset")
rc_fig = go.Figure([go.Bar(x=risk_contrib_df.index, y=risk_contrib_df.values)])
rc_fig.update_layout(template="plotly_dark", title="Risk Contribution", xaxis_title="Ticker", yaxis_title="Risk")
st.plotly_chart(rc_fig, use_container_width=True)

# ------------------ CORRELATION HEATMAP ------------------ #
st.subheader("Correlation Matrix")
corr = returns[TICKERS].corr()
corr_fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale='Blues'))
corr_fig.update_layout(template="plotly_dark")
st.plotly_chart(corr_fig, use_container_width=True)

# ------------------ EXPORT ------------------ #
st.download_button("Export Metrics", data=port_returns.to_csv().encode('utf-8'), file_name='portfolio_returns.csv', mime='text/csv')
