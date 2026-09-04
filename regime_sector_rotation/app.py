import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import requests
from datetime import datetime

# --- Page Setup ---
st.set_page_config(
    page_title="Quant Regime Rotation Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Modern Minimalist & Cohesive Pitch-Black Aesthetics ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Font and Background Overrides */
    html, body, [class*="css"], .stApp, div[data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* 2. Top Header and Main Section styling */
    header[data-testid="stHeader"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #111111 !important;
    }
    
    div[data-testid="stHeaderDecoration"] {
        background-image: linear-gradient(to right, #ef4444, #10b981) !important;
        height: 3px !important;
    }
    
    .main {
        background-color: #000000 !important;
    }
    
    /* 3. Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px !important;
    }
    
    /* 4. Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #111111 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cccccc !important;
        font-size: 13px !important;
    }
    
    /* 5. Premium Minimalist Metric Cards */
    .metric-card {
        background-color: #080808 !important;
        border: 1px solid #1a1a1a !important;
        border-radius: 4px !important;
        padding: 20px !important;
        text-align: left !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        transition: border-color 0.2s ease !important;
    }
    
    .metric-card:hover {
        border-color: #333333 !important;
    }
    
    .metric-value {
        font-size: 26px !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.5px !important;
        margin-top: 5px !important;
    }
    
    .metric-label {
        font-size: 10px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: #888888 !important;
    }
    
    /* 6. Button overrides */
    div.stButton > button {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #222222 !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("Regime-Aware Sector Rotation Terminal")

current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(current_dir, "data", "trade_log.csv")
BACKTEST_EQ_PATH = os.path.join(current_dir, "data", "backtest_equity.csv")
MANIFEST_PATH = os.path.join(current_dir, "data", "run_manifest.json")

run_manifest = None
if os.path.exists(MANIFEST_PATH):
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
            run_manifest = json.load(manifest_file)
    except (OSError, ValueError):
        run_manifest = None

if run_manifest:
    mode = str(run_manifest.get("data_mode", "unknown")).upper()
    generated = run_manifest.get("generated_at", "unknown time")
    execution_end = run_manifest.get("execution_end", "unknown")
    st.info(
        f"RESEARCH ARTIFACTS · {mode} DATA · generated {generated} · "
        f"last simulated execution {execution_end}. This is not live brokerage execution."
    )
else:
    st.warning(
        "LEGACY ARTIFACTS · These committed charts predate the integrity refactor. "
        "They are historical backtest/simulation outputs, not a live or paper-trading record. "
        "Run the real-data pipeline to generate a provenance manifest."
    )

# Dict mapping HMM states to structured, technical visual styles & details
REGIME_DETAILS = {
    0: {"name": "State 0: Low Volatility (Bullish Expansion)", "desc": "Optimal equity rotation window", "color": "#10b981", "risk": "100%"},
    1: {"name": "State 1: Moderate Volatility (Transition)", "desc": "Tactical exposure containment active", "color": "#f59e0b", "risk": "80%"},
    2: {"name": "State 2: High Volatility (Crisis Overlay)", "desc": "Significant defensive cash/TLT overlay", "color": "#ef4444", "risk": "50%"},
    3: {"name": "State 3: Extreme Volatility (Severe Stress)", "desc": "Complete risk-off defensive stance", "color": "#7f1d1d", "risk": "0%"}
}

# Check if trade log exists
if not os.path.exists(LOG_PATH):
    st.warning("No trade database found. Please execute the strategy pipeline to initialize tracking.")
    if st.button("Initialize & Run Strategy Pipeline"):
        with st.spinner("Executing pipeline ingestion and model training..."):
            from main import execute_orchestration
            try:
                execute_orchestration(weeks=26, force_refresh=True)
                st.success("Strategy pipeline completed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")
else:
    # Load and parse trade logs
    df = pd.read_csv(LOG_PATH)
    
    # 1. Parse Allocation Rows
    df_alloc = df[df['Action'] == 'ALLOCATION'].copy()
    
    # 2. Parse Equity Metric Rows
    df_equity = df[(df['Asset'] == 'PORTFOLIO_METRIC') & (df['Action'] == 'EQUITY')].copy()
    
    # 3. Parse Trade Ledger Rows
    df_ledger = df[df['Action'].isin(['BUY', 'SELL'])].copy()
    
    # 4. Parse Regime Metrics Rows
    df_regime = df[(df['Asset'] == 'REGIME_METRIC') & (df['Action'] == 'STATE')].copy()
    
    # 5. Load full backtest equity curve and calculate advanced metrics
    df_be = None
    adv_metrics = None
    if os.path.exists(BACKTEST_EQ_PATH):
        df_be = pd.read_csv(BACKTEST_EQ_PATH)
        df_be['Date'] = pd.to_datetime(df_be['Date'])
        df_be = df_be.sort_values('Date')
        
        if not df_be.empty:
            strat_ret_be = df_be['Strategy'].pct_change().dropna()
            ann_factor = 252  # Backtest artifacts are daily mark-to-market returns.
            
            strat_vol = strat_ret_be.std() * np.sqrt(ann_factor)
            strat_ann_ret = strat_ret_be.mean() * ann_factor
            strat_sharpe = strat_ann_ret / strat_vol if strat_vol != 0 else 0
            
            strat_downside = strat_ret_be[strat_ret_be < 0]
            strat_downside_std = strat_downside.std() * np.sqrt(ann_factor) if len(strat_downside) > 0 else 0
            strat_sortino = strat_ann_ret / strat_downside_std if strat_downside_std != 0 else 0
            
            strat_peak = df_be['Strategy'].cummax()
            strat_dd = (df_be['Strategy'] / strat_peak) - 1
            strat_max_dd = strat_dd.min() * 100
            
            adv_metrics = {
                'sharpe': strat_sharpe,
                'sortino': strat_sortino,
                'max_dd': strat_max_dd,
                'vol': strat_vol * 100
            }
    
    # --- Live System Signals Header (CURRENT / LATEST DATE) ---
    if not df_alloc.empty:
        latest_date = df_alloc['Timestamp'].max()
        
        # Get active regime & risk scalar for latest date
        latest_regime_row = df_regime[df_regime['Timestamp'] == latest_date]
        if not latest_regime_row.empty:
            latest_regime = int(latest_regime_row.iloc[0]['Vol'])
            latest_risk_scalar = float(latest_regime_row.iloc[0]['CostBasis'])
        else:
            latest_regime = 0
            latest_risk_scalar = 1.0
            
        # Get target sector holdings for latest date
        df_latest_alloc = df_alloc[df_alloc['Timestamp'] == latest_date].copy()
        active_holdings = df_latest_alloc[df_latest_alloc['Vol'] > 0]
        holdings_str = " | ".join([f"{row['Asset']}: {row['Vol']*100:.1f}%" for _, row in active_holdings.iterrows()])
        if not holdings_str:
            holdings_str = "100.0% Cash / Defensive Assets"
            
        # Display elegant full-width signal status card
        st.markdown(f"""
        <div style='background-color: #080808; border: 1px solid #1a1a1a; border-radius: 4px; padding: 18px; margin-bottom: 25px;'>
            <div style='font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #888888; margin-bottom: 12px;'>
                LIVE SYSTEM SIGNALS & ACTIVE POSITIONING (WEEK ENDING: {latest_date})
            </div>
            <div style='display: flex; flex-wrap: wrap; gap: 30px;'>
                <div style='flex: 1.2; min-width: 250px;'>
                    <div style='font-size: 11px; color: #888888; font-weight: 500; text-transform: uppercase;'>Current Market Regime</div>
                    <div style='font-size: 18px; font-weight: 700; color: {REGIME_DETAILS.get(latest_regime, {}).get("color", "#ffffff")}; margin-top: 4px;'>
                        {REGIME_DETAILS.get(latest_regime, {}).get("name", f"State {latest_regime}")}
                    </div>
                    <div style='font-size: 13px; color: #cccccc; margin-top: 2px;'>
                        {REGIME_DETAILS.get(latest_regime, {}).get("desc", "N/A")}
                    </div>
                </div>
                <div style='flex: 0.8; min-width: 150px;'>
                    <div style='font-size: 11px; color: #888888; font-weight: 500; text-transform: uppercase;'>System Risk Exposure</div>
                    <div style='font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 4px;'>
                        {latest_risk_scalar * 100:.0f}%
                    </div>
                    <div style='font-size: 11px; color: #888888;'>
                        {'Risk-On Active Sector Alloc' if latest_risk_scalar > 0.0 else 'Risk-Off Complete Defense'}
                    </div>
                </div>
                <div style='flex: 2; min-width: 300px;'>
                    <div style='font-size: 11px; color: #888888; font-weight: 500; text-transform: uppercase;'>Current Predicted Holdings</div>
                    <div style='font-size: 15px; font-weight: 600; color: #10b981; margin-top: 8px; font-family: monospace;'>
                        {holdings_str}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Sidebar Execution Controls ---
    st.sidebar.header("System Execution Panel")
    st.sidebar.markdown("---")
    
    st.sidebar.write("**System Status**: Operational")
    st.sidebar.write(f"**Executed Trades**: {len(df_ledger)}")
    st.sidebar.write(f"**Tracking Horizon**: {df['Timestamp'].nunique()} weeks")
    
    sim_weeks = st.sidebar.slider("Execution Simulation Weeks", min_value=12, max_value=104, value=26)
    
    if st.sidebar.button("Run Live Strategy Pipeline"):
        with st.spinner("Downloading live market data and retraining models..."):
            from main import execute_orchestration
            try:
                execute_orchestration(weeks=sim_weeks, force_refresh=True)
                st.success("Pipeline completed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")
                
    st.sidebar.markdown("---")
    mtime = os.path.getmtime(LOG_PATH)
    last_run = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.write(f"**Last Sync Date**: {last_run}")

    # --- KPI Cards Row ---
    if not df_equity.empty:
        # Latest equity values
        latest_row = df_equity.iloc[-1]
        first_row = df_equity.iloc[0]
        
        current_eq = latest_row['Vol']  # Strategy Equity stored in Vol
        spy_eq = latest_row['CostBasis']  # SPY Equity stored in CostBasis
        
        strat_ret = ((current_eq / first_row['Vol']) - 1) * 100
        spy_ret = ((spy_eq / first_row['CostBasis']) - 1) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>PORTFOLIO NET EQUITY</div>
                <div class='metric-value' style='color: #ffffff;'>${current_eq:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>SPY BENCHMARK VALUE</div>
                <div class='metric-value' style='color: #ffffff;'>${spy_eq:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            color = "#10b981" if strat_ret >= 0 else "#ef4444"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>STRATEGY TOTAL RETURN</div>
                <div class='metric-value' style='color: {color};'>{strat_ret:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            color = "#10b981" if spy_ret >= 0 else "#ef4444"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>SPY BENCHMARK RETURN</div>
                <div class='metric-value' style='color: {color};'>{spy_ret:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        if adv_metrics is not None:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            k5, k6, k7, k8 = st.columns(4)
            with k5:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>ANNUAL SHARPE RATIO</div>
                    <div class='metric-value' style='color: #ffffff;'>{adv_metrics['sharpe']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k6:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>ANNUAL SORTINO RATIO</div>
                    <div class='metric-value' style='color: #ffffff;'>{adv_metrics['sortino']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k7:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>MAX HISTORICAL DRAWDOWN</div>
                    <div class='metric-value' style='color: #ef4444;'>{adv_metrics['max_dd']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with k8:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>ANNUALIZED VOLATILITY</div>
                    <div class='metric-value' style='color: #ffffff;'>{adv_metrics['vol']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Layout Columns ---
    col_left, col_right = st.columns([1, 1])
    
    # --- Left Column: Dynamic Capital Allocation History Explorer ---
    with col_left:
        st.subheader("Capital Allocation Explorer")
        if not df_alloc.empty:
            # Sorted unique timestamps for selectbox
            unique_dates = sorted(df_alloc['Timestamp'].unique(), reverse=True)
            
            selected_date = st.selectbox(
                "Select Rebalance Date to View Allocation:",
                options=unique_dates,
                index=0,
                key="allocation_date_selector"
            )
            
            # Fetch active allocation for selected date
            df_curr_alloc = df_alloc[df_alloc['Timestamp'] == selected_date].copy()
            
            # Fetch historical HMM details for selected date
            sel_regime_row = df_regime[df_regime['Timestamp'] == selected_date]
            if not sel_regime_row.empty:
                sel_regime = int(sel_regime_row.iloc[0]['Vol'])
                sel_risk_scalar = float(sel_regime_row.iloc[0]['CostBasis'])
            else:
                sel_regime = 0
                sel_risk_scalar = 1.0
                
            # Render descriptive sub-banner
            st.markdown(f"""
            <div style='background-color: #050505; border: 1px solid #111111; padding: 12px; border-radius: 4px; margin-bottom: 15px;'>
                <span style='color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase;'>Historical Regime:</span>
                <span style='color: {REGIME_DETAILS.get(sel_regime, {}).get("color", "#ffffff")}; font-weight: 600; font-size: 13px; margin-left: 5px;'>
                    {REGIME_DETAILS.get(sel_regime, {}).get("name", f"State {sel_regime}")}
                </span>
                <span style='color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; margin-left: 20px;'>Risk Exposure:</span>
                <span style='color: #ffffff; font-weight: 600; font-size: 13px; margin-left: 5px;'>
                    {sel_risk_scalar * 100:.0f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Sum up allocated weights
            allocated_weight = df_curr_alloc['Vol'].sum()
            cash_weight = max(0.0, 1.0 - allocated_weight)
            
            # Prepare data for pie chart
            pie_data = []
            for _, r in df_curr_alloc.iterrows():
                if r['Vol'] > 0:
                    pie_data.append({"Asset": r['Asset'], "Weight": r['Vol']})
            if cash_weight > 0.001:
                pie_data.append({"Asset": "CASH", "Weight": cash_weight})
                
            df_pie = pd.DataFrame(pie_data)
            
            # Render Donut Chart
            # Emerald theme gradient/colors
            custom_colors = ['#10b981', '#34d399', '#059669', '#047857', '#065f46', '#064e3b', '#111827', '#1f2937', '#374151']
            fig_pie = px.pie(
                df_pie, 
                values='Weight', 
                names='Asset', 
                hole=0.5,
                color_discrete_sequence=custom_colors
            )
            fig_pie.update_layout(
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font=dict(color='#ffffff', family='Inter'),
                showlegend=True,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#000000', width=1.5))
            )
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No active allocation metrics logged yet.")

    # --- Right Column: Performance Equity curve vs SPY (FULL STRATEGY BACKTEST) ---
    with col_right:
        st.subheader("Cumulative Wealth Performance")
        
        if os.path.exists(BACKTEST_EQ_PATH):
            df_be = pd.read_csv(BACKTEST_EQ_PATH)
            df_be['Date'] = pd.to_datetime(df_be['Date'])
            df_be = df_be.sort_values('Date')
            
            # Scale both paths to start at exactly $100,000.00 to represent capital growth paths
            df_be['Strategy_Scaled'] = (df_be['Strategy'] / df_be['Strategy'].iloc[0]) * 100000.0
            df_be['Benchmark_Scaled'] = (df_be['Benchmark'] / df_be['Benchmark'].iloc[0]) * 100000.0
            comparison_column = 'SPY' if 'SPY' in df_be.columns else 'Benchmark'
            df_be['SPY_Scaled'] = (df_be[comparison_column] / df_be[comparison_column].iloc[0]) * 100000.0
            
            # Render Line Chart
            fig_line = go.Figure()
            
            # Strategy Equity Path (Full History)
            fig_line.add_trace(go.Scatter(
                x=df_be['Date'],
                y=df_be['Strategy_Scaled'],
                mode='lines',
                name='Strategy (Net)',
                line=dict(color='#10b981', width=2.5) # Clean emerald green
            ))
            
            # SPY Benchmark Path (Full History)
            fig_line.add_trace(go.Scatter(
                x=df_be['Date'],
                y=df_be['SPY_Scaled'],
                mode='lines',
                name='SPY Benchmark',
                line=dict(color='#ffffff', width=1.5, dash='dot') # Clean minimalist white dotted
            ))
            
            fig_line.update_layout(
                title="Daily Mark-to-Market Strategy vs. SPY",
                xaxis_title="Date",
                yaxis_title="Equity Value ($)",
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font=dict(color='#ffffff', family='Inter'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(showgrid=True, gridcolor='#111111', linecolor='#222222'),
                yaxis=dict(showgrid=True, gridcolor='#111111', linecolor='#222222'),
                margin=dict(t=50, b=20, l=10, r=10),
                height=355
            )
            
            st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No historical backtest equity log found.")
            
    # --- Middle Section: Historical Sector Weightings Rotation (FULL BACKTEST) ---
    st.write("---")
    st.subheader("Historical Sector Weightings Rotation")
    
    BACKTEST_WEIGHTS_PATH = os.path.join(current_dir, "data", "backtest_weights.csv")
    if os.path.exists(BACKTEST_WEIGHTS_PATH):
        df_bw = pd.read_csv(BACKTEST_WEIGHTS_PATH)
        df_bw['Date'] = pd.to_datetime(df_bw['Date'])
        df_bw = df_bw.sort_values('Date')
        
        # Get list of sector columns (excluding 'Date')
        sector_cols = [col for col in df_bw.columns if col != 'Date']
        
        # Multiselect for sectors to display
        selected_sectors = st.multiselect(
            "Select Sectors to View Historical Weightings (2014-Present):",
            options=sector_cols,
            default=['XLK', 'XLF', 'XLY', 'TLT']
        )
        
        if selected_sectors:
            fig_weights = go.Figure()
            
            # Palette for the weight chart
            weight_colors = {
                'XLK': '#10b981', 'XLF': '#3b82f6', 'XLY': '#f59e0b', 'XLE': '#ef4444', 
                'XLV': '#ec4899', 'XLI': '#8b5cf6', 'XLP': '#6b7280', 'XLU': '#14b8a6', 
                'XLB': '#f97316', 'XLC': '#84cc16', 'XLRE': '#06b6d4', 'TLT': '#ffffff'
            }
            
            for col in selected_sectors:
                color = weight_colors.get(col, None)
                fig_weights.add_trace(go.Scatter(
                    x=df_bw['Date'],
                    y=df_bw[col] * 100.0, # percentage weight
                    mode='lines',
                    name=col,
                    line=dict(width=2, color=color)
                ))
                
            fig_weights.update_layout(
                title="Historical Model Allocation Weights over the Backtest Horizon (2014-Present)",
                xaxis_title="Date",
                yaxis_title="Allocation Weight (%)",
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font=dict(color='#ffffff', family='Inter'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(showgrid=True, gridcolor='#111111', linecolor='#222222'),
                yaxis=dict(showgrid=True, gridcolor='#111111', linecolor='#222222', range=[-5, 105]),
                margin=dict(t=50, b=20, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_weights, width="stretch")
        else:
            st.info("Select one or more sectors from the multiselect dropdown above to view weight history.")
    else:
        st.info("No historical backtest weightings log found.")
        
    st.write("---")
    
    # --- Bottom Section: Transaction Ledger ---
    st.subheader("Transaction Ledger")
    
    if not df_ledger.empty:
        df_ledger_display = df_ledger.copy()
        df_ledger_display = df_ledger_display.rename(columns={
            "Timestamp": "Trade Date",
            "Asset": "Asset Ticker",
            "Action": "Trade Action",
            "Vol": "Shares Traded",
            "CostBasis": "Execution Price ($)"
        })
        
        df_ledger_display = df_ledger_display.sort_values("Trade Date", ascending=False)
        
        search_term = st.text_input("Filter ledger by ticker or action (e.g. XLK, BUY, SELL)", "")
        
        if search_term:
            filtered_df = df_ledger_display[
                df_ledger_display['Asset Ticker'].str.contains(search_term, case=False) |
                df_ledger_display['Trade Action'].str.contains(search_term, case=False)
            ]
            st.dataframe(filtered_df, width="stretch", hide_index=True)
        else:
            st.dataframe(df_ledger_display, width="stretch", hide_index=True)
            
    else:
        st.info("No trade transactions executed yet. This table will populate when portfolio rebalancing commands are executed.")
