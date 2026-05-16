from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

import generate_data
from data_pipeline import run_pipeline, apply_filters, join_data
from analytics_engine import (
    executive_kpis,
    region_summary,
    build_forecasts,
    sales_metrics,
    marketing_metrics,
    advertising_metrics,
    hr_metrics,
    scenario_simulation,
    linear_regression_forecast,
    linear_regression_simulation,
    insight_text,
)


st.set_page_config(
    page_title="Cyber Nova BI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# STREAMLIT NATIVE CSS
STREAMLIT_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root{
  --bg:#07101e;
  --panel:#0c263f;
  --panel2:#123a5b;
  --panel3:#061727;
  --line:rgba(105,210,255,.48);
  --line-soft:rgba(105,210,255,.22);
  --text:#f5fbff;
  --soft:#c8e8f8;
  --muted:#8bb4cc;
  --cyan:#6bdcff;
  --blue:#2f97ff;
  --green:#37f084;
  --amber:#ffc266;
  --red:#ff6677;
  --purple:#a78bfa;
}

html, body, [class*="css"]{
  font-family:'Inter','Segoe UI',Arial,sans-serif!important;
}

.stApp{
  color:var(--text)!important;
  background:
    radial-gradient(circle at 12% 0%, rgba(47,151,255,.18), transparent 28%),
    radial-gradient(circle at 92% 10%, rgba(107,220,255,.12), transparent 24%),
    linear-gradient(135deg,#06101e 0%,#071f35 46%,#020812 100%)!important;
}

[data-testid="stHeader"]{
  background:rgba(0,0,0,0)!important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{
  display:none!important;
  visibility:hidden!important;
}

.block-container{
  max-width:1780px!important;
  padding-top:.30rem!important;
  padding-bottom:1.40rem!important;
  padding-left:.75rem!important;
  padding-right:.75rem!important;
}

/* Streamlit layout spacing only. Do not turn Streamlit blocks into CSS grids. */
div[data-testid="stVerticalBlock"]{gap:.30rem!important;}
div[data-testid="column"]{padding-left:.16rem!important;padding-right:.16rem!important;}
.st-key-visual_grid div[data-testid="column"]{padding-left:.18rem!important;padding-right:.18rem!important;}
.st-key-main_workspace{margin-top:8px!important;}

/* TOP BAR */
.st-key-top_bar{
  border:1px solid var(--line)!important;
  border-radius:0!important;
  padding:8px!important;
  margin-bottom:10px!important;
  background:linear-gradient(180deg,rgba(15,45,72,.92),rgba(6,22,38,.94))!important;
  box-shadow:0 18px 42px rgba(0,0,0,.32), inset 0 0 26px rgba(107,220,255,.07)!important;
}

.st-key-title_cell,
.st-key-navigation_cell,
.st-key-settings_cell,
.st-key-status_cell{
  min-height:62px!important;
  border:1px solid rgba(107,220,255,.32)!important;
  border-radius:0!important;
  background:linear-gradient(180deg,rgba(18,55,86,.58),rgba(4,16,29,.62))!important;
  box-shadow:inset 0 0 18px rgba(107,220,255,.06)!important;
  padding:6px 8px!important;
}

.st-key-title_cell .stMarkdown,
.st-key-status_cell .stMarkdown{
  margin:0!important;
}

.title-main{
  color:#91e3ff!important;
  font-size:.98rem!important;
  font-weight:900!important;
  line-height:1!important;
  letter-spacing:.06em!important;
  white-space:nowrap!important;
  padding-top:8px!important;
}
.title-sub{
  color:#c8efff!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.52rem!important;
  letter-spacing:.18em!important;
  text-transform:uppercase!important;
  margin-top:4px!important;
  opacity:.92!important;
}

.st-key-navigation_cell div[data-testid="stHorizontalBlock"]{gap:.25rem!important;}
[class*="st-key-nav_idle_"], [class*="st-key-nav_active_"]{min-height:44px!important;}
[class*="st-key-nav_idle_"] button,
[class*="st-key-nav_active_"] button{
  min-height:42px!important;
  height:42px!important;
  border-radius:0!important;
  border:1px solid rgba(107,220,255,.34)!important;
  box-shadow:none!important;
  padding:2px 5px!important;
  white-space:pre-line!important;
}
[class*="st-key-nav_idle_"] button{
  background:rgba(5,17,31,.42)!important;
  color:#d7f3ff!important;
}
[class*="st-key-nav_active_"] button{
  background:linear-gradient(180deg,rgba(91,179,255,.48),rgba(39,104,157,.44))!important;
  border-color:rgba(142,226,255,.88)!important;
  box-shadow:inset 0 -2px 0 var(--cyan), inset 0 0 18px rgba(107,220,255,.18)!important;
  color:#fff!important;
}
[class*="st-key-nav_idle_"] button p,
[class*="st-key-nav_active_"] button p{
  font-size:.66rem!important;
  line-height:1.08!important;
  margin:0!important;
  text-align:center!important;
  font-weight:750!important;
  text-transform:uppercase!important;
}

.st-key-settings_cell button,
.st-key-status_cell button{
  min-height:38px!important;
  height:38px!important;
  border-radius:999px!important;
  border:1px solid rgba(107,220,255,.40)!important;
  background:rgba(9,26,44,.70)!important;
  color:#eefbff!important;
  font-size:.66rem!important;
}
.st-key-settings_cell div[data-testid="stVerticalBlock"]{align-items:center!important;justify-content:center!important;}
.st-key-settings_cell [data-testid="stPopover"]{display:flex!important;justify-content:center!important;}
.st-key-settings_cell [data-testid="stPopover"] button{
  min-height:36px!important;
  height:36px!important;
  width:36px!important;
  padding:0!important;
  font-size:1.20rem!important;
  font-weight:800!important;
  border-radius:50%!important;
}
.settings-trigger-wrap{
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
  gap:1px!important;
  margin-top:1px!important;
}
.settings-caption{
  display:block!important;
  margin-top:4px!important;
  width:100%!important;
  text-align:center!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.48rem!important;
  color:var(--soft)!important;
  letter-spacing:.12em!important;
  text-transform:uppercase!important;
  line-height:1!important;
}
.live-cell{
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  min-height:38px!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.72rem!important;
  color:var(--green)!important;
  font-weight:800!important;
  white-space:nowrap!important;
}
.live-dot{
  display:inline-block!important;
  width:11px!important;
  height:11px!important;
  border-radius:50%!important;
  margin-right:8px!important;
  background:var(--green)!important;
  box-shadow:0 0 18px var(--green)!important;
}

/* KPI ROW */
.st-key-kpi_row{
  position:relative!important;
  top:-14px!important;
  margin:6px 0 24px!important;
  padding:0!important;
  border:none!important;
  background:transparent!important;
}
.st-key-kpi_row div[data-testid="column"]{padding-left:.38rem!important;padding-right:.38rem!important;}
.kpi-card{
  position:relative!important;
  overflow:hidden!important;
  text-align:center!important;
  min-height:108px!important;
  border-radius:8px!important;
  border:1px solid rgba(189,236,255,.62)!important;
  background:linear-gradient(115deg,rgba(28,72,105,.88),rgba(15,39,64,.78) 50%,rgba(120,165,190,.30))!important;
  box-shadow:inset 0 0 24px rgba(155,226,255,.16),0 10px 24px rgba(0,0,0,.22)!important;
  padding:12px 16px!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
}
.kpi-card:after{
  content:""!important;
  position:absolute!important;
  left:0!important;right:0!important;bottom:0!important;
  height:2px!important;
  background:linear-gradient(90deg,var(--blue),var(--cyan),rgba(107,220,255,.05))!important;
}
.kpi-id{
  display:block!important;
  text-align:center!important;
  color:#fff!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.68rem!important;
  letter-spacing:.08em!important;
  margin-bottom:2px!important;
}
.kpi-label{
  text-align:center!important;
  color:#f5fbff!important;
  font-size:.76rem!important;
  letter-spacing:.01em!important;
  font-weight:750!important;
}
.kpi-value{
  color:#fff!important;
  text-align:center!important;
  font-size:1.48rem!important;
  font-weight:900!important;
  letter-spacing:-.04em!important;
  margin-top:7px!important;
}
.kpi-note{font-size:.70rem!important;color:#9fdcf4!important;margin-top:2px!important;text-align:center!important;}
.arrow-up{color:var(--green)!important;font-weight:900!important;}
.arrow-down{color:var(--red)!important;font-weight:900!important;}
.arrow-watch{color:var(--amber)!important;font-weight:900!important;}
.arrow-info{color:var(--cyan)!important;font-weight:900!important;}

/*PANELS*/
.st-key-filter_panel,
.st-key-visual_1,
.st-key-visual_2,
.st-key-visual_3,
.st-key-visual_4,
.st-key-main_visual,
.st-key-insight_panel,
.st-key-forecast_main,
.st-key-forecast_card_1,
.st-key-forecast_card_2,
.st-key-forecast_card_3,
.st-key-report_main,
.st-key-report_source_panel,
.st-key-report_history_panel,
.st-key-report_right{
  border:1px solid rgba(107,220,255,.50)!important;
  border-radius:0!important;
  background:linear-gradient(180deg,rgba(12,37,61,.78),rgba(4,15,28,.82))!important;
  box-shadow:inset 0 0 30px rgba(107,220,255,.06),0 10px 26px rgba(0,0,0,.22)!important;
  overflow:hidden!important;
}

.st-key-filter_panel{min-height:560px!important;padding:12px!important;}
.st-key-main_visual{min-height:560px!important;padding:10px!important;}
.st-key-insight_panel{min-height:560px!important;padding:10px!important;overflow:auto!important;}
.st-key-visual_1,.st-key-visual_2,.st-key-visual_3,.st-key-visual_4{min-height:276px!important;padding:10px!important;}
.st-key-forecast_main{min-height:548px!important;padding:8px!important;}
.st-key-forecast_card_1,.st-key-forecast_card_2,.st-key-forecast_card_3{min-height:170px!important;padding:7px!important;}
.st-key-report_main{min-height:448px!important;padding:12px!important;}
.st-key-report_source_panel{min-height:214px!important;padding:8px!important;}
.st-key-report_history_panel{min-height:214px!important;padding:10px!important;}
.st-key-report_right{min-height:448px!important;padding:12px!important;}

.st-key-visual_stack_left div[data-testid="stVerticalBlock"],
.st-key-visual_stack_right div[data-testid="stVerticalBlock"],
.st-key-forecast_side_stack div[data-testid="stVerticalBlock"],
.st-key-report_context_stack div[data-testid="stVerticalBlock"]{
  gap:.62rem!important;
}

.panel-heading,
.panel-heading.center{
  text-align:center!important;
  color:#f5fbff!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.80rem!important;
  letter-spacing:.05em!important;
  text-transform:uppercase!important;
  margin-bottom:8px!important;
}

/*STREAMLIT CONTROLS*/
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSlider"] label,
div[data-testid="stToggle"] label,
div[data-testid="stRadio"] label{
  color:#dff6ff!important;
  font-weight:750!important;
  font-size:.74rem!important;
}

.st-key-filter_panel [data-testid="stSelectbox"]{margin-bottom:.32rem!important;}
div[data-baseweb="select"] > div,
.stTextInput input,
textarea{
  background:linear-gradient(180deg,rgba(34,67,94,.72),rgba(7,22,38,.84))!important;
  border:1px solid rgba(107,220,255,.35)!important;
  color:var(--text)!important;
  border-radius:6px!important;
  min-height:42px!important;
  font-size:.76rem!important;
}
textarea{min-height:auto!important;}

.stDownloadButton button,
.stButton button{
  border:1px solid rgba(107,220,255,.42)!important;
  background:linear-gradient(180deg,rgba(34,100,150,.66),rgba(8,31,54,.80))!important;
  color:#f5fbff!important;
  border-radius:6px!important;
  font-weight:750!important;
}

/*FILTER LOGS, REPORT CARDS, INSIGHTS */
.filter-log-box,
.report-history-card,
.distribution-card,
.source-node-card,
.story-card,
.ai-item{
  border:1px solid rgba(107,220,255,.22)!important;
  background:rgba(4,14,26,.58)!important;
  border-radius:8px!important;
}
.filter-log-box{margin-top:12px!important;min-height:104px!important;padding:9px!important;}
.filter-log-title{font-family:'IBM Plex Mono',monospace!important;font-size:.76rem!important;color:#f5fbff!important;text-align:center!important;margin-bottom:6px!important;text-transform:uppercase!important;}
.filter-log-line{font-family:'IBM Plex Mono',monospace!important;font-size:.56rem!important;color:#9fdcf4!important;line-height:1.35!important;}
.forecast-control-box{margin-top:12px!important;border:1px solid rgba(107,220,255,.24)!important;background:rgba(4,14,26,.55)!important;border-radius:8px!important;padding:8px!important;}
.forecast-control-title{font-family:'IBM Plex Mono',monospace!important;font-size:.62rem!important;color:var(--cyan)!important;letter-spacing:.10em!important;text-transform:uppercase!important;margin-bottom:4px!important;text-align:center!important;}
.forecast-summary-line{font-family:'IBM Plex Mono',monospace!important;font-size:.55rem!important;color:#a7e6ff!important;line-height:1.35!important;}

.report-card-title{
  color:var(--cyan)!important;
  font-family:'IBM Plex Mono',monospace!important;
  font-size:.68rem!important;
  letter-spacing:.08em!important;
  text-transform:uppercase!important;
  margin-bottom:8px!important;
}
.report-history-card,.distribution-card,.source-node-card{
  color:var(--soft)!important;
  padding:8px 9px!important;
  margin-bottom:7px!important;
  font-size:.68rem!important;
  line-height:1.25!important;
}

.insight-panel-inner{height:auto!important;min-height:100%!important;}
.story-card{padding:8px!important;margin-bottom:8px!important;}
.status-stack{display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;min-height:46px!important;}
.status-title{font-family:'IBM Plex Mono',monospace!important;font-size:.52rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:#d7f3ff!important;margin-bottom:3px!important;}
.st-key-forecast_control_bar{border:1px solid rgba(107,220,255,.46)!important;border-radius:0!important;background:linear-gradient(180deg,rgba(12,37,61,.78),rgba(4,15,28,.82))!important;box-shadow:inset 0 0 30px rgba(107,220,255,.05),0 8px 20px rgba(0,0,0,.20)!important;padding:10px 12px!important;margin:0 0 10px!important;}
.st-key-forecast_control_bar div[data-testid='stHorizontalBlock']{gap:.55rem!important;}
.forecast-control-caption{font-family:'IBM Plex Mono',monospace!important;font-size:.68rem!important;color:var(--cyan)!important;letter-spacing:.10em!important;text-transform:uppercase!important;margin-bottom:8px!important;text-align:center!important;}
.forecast-summary-strip{margin-top:8px!important;border:1px solid rgba(107,220,255,.22)!important;border-radius:6px!important;background:rgba(4,14,26,.56)!important;padding:8px 10px!important;color:var(--soft)!important;font-size:.66rem!important;line-height:1.35!important;}
.story-label{color:var(--cyan)!important;font-family:'IBM Plex Mono',monospace!important;font-size:.50rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;margin-bottom:3px!important;}
.story-text{font-size:.67rem!important;line-height:1.28!important;color:var(--text)!important;font-weight:650!important;}
.score-line{height:7px!important;border-radius:999px!important;background:rgba(255,255,255,.10)!important;overflow:hidden!important;margin-top:6px!important;}
.score-fill{height:100%!important;background:linear-gradient(90deg,var(--red),var(--amber),var(--green))!important;}
.ai-list{display:grid!important;gap:6px!important;}
.ai-item{border-left:2px solid var(--cyan)!important;padding:6px 8px!important;color:var(--soft)!important;font-size:.63rem!important;line-height:1.24!important;}

/* PLOTLY*/
[data-testid="stPlotlyChart"]{
  margin:0!important;
  border-radius:0!important;
  overflow:hidden!important;
}
.js-plotly-plot .plotly .legend text{font-size:9px!important;}
.svg-container text{font-family:Inter,Segoe UI,Arial!important;}

/* MOVING BAR*/
.st-key-moving_info_bar{
  margin-top:12px!important;
  margin-bottom:22px!important;
  min-height:66px!important;
  border:1px solid rgba(107,220,255,.50)!important;
  border-radius:0!important;
  background:rgba(5,14,26,.65)!important;
  overflow:hidden!important;
}
.ticker{position:relative!important;overflow:hidden!important;height:58px!important;background:linear-gradient(90deg,rgba(6,24,40,.96),rgba(9,38,59,.96))!important;}
.ticker-track{position:absolute!important;white-space:nowrap!important;animation:tickerMove 96s linear infinite!important;padding:17px 0!important;color:var(--soft)!important;font-size:.76rem!important;line-height:1.25!important;}
.ticker-track:hover{animation-play-state:paused!important;}
.ticker-item{margin-right:54px!important;}
.ticker-label{color:var(--cyan)!important;font-family:'IBM Plex Mono',monospace!important;font-weight:800!important;letter-spacing:.03em!important;}
@keyframes tickerMove{0%{transform:translateX(100%)}100%{transform:translateX(-100%)}}

hr{border-color:rgba(86,190,255,.14)!important;}

@media(max-width:1200px){
  .block-container{padding-left:.5rem!important;padding-right:.5rem!important;}
  .st-key-filter_panel,.st-key-main_visual,.st-key-insight_panel,.st-key-forecast_main,.st-key-report_main,.st-key-report_right{min-height:auto!important;}
  .st-key-visual_1,.st-key-visual_2,.st-key-visual_3,.st-key-visual_4,.st-key-forecast_card_1,.st-key-forecast_card_2,.st-key-forecast_card_3{min-height:auto!important;}
}
"""

st.markdown(f"<style>{STREAMLIT_CSS}</style>", unsafe_allow_html=True)


# SESSION STATE

if "data_initialised" not in st.session_state:
    generate_data.initialise_data()
    st.session_state.data_initialised = True
    st.session_state.tick_count = 0
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    st.session_state.live_log = []

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

if "live_mode" not in st.session_state:
    st.session_state.live_mode = True

if "refresh_speed" not in st.session_state:
    st.session_state.refresh_speed = 30

if "rows_per_tick" not in st.session_state:
    st.session_state.rows_per_tick = 5

if "forecast_mode" not in st.session_state:
    st.session_state.forecast_mode = "Forecast"
if "forecast_view" not in st.session_state:
    st.session_state.forecast_view = "Enterprise"
if "forecast_department" not in st.session_state:
    st.session_state.forecast_department = "Overview"
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 14
if "forecast_metric" not in st.session_state:
    st.session_state.forecast_metric = "Revenue"
if "forecast_hr_department" not in st.session_state:
    st.session_state.forecast_hr_department = "All"
if "sim_demand_shift" not in st.session_state:
    st.session_state.sim_demand_shift = 8
if "sim_price_shift" not in st.session_state:
    st.session_state.sim_price_shift = 3
if "sim_spend_shift" not in st.session_state:
    st.session_state.sim_spend_shift = 5
if "sim_capacity_shift" not in st.session_state:
    st.session_state.sim_capacity_shift = 2
if "simulation_runs" not in st.session_state:
    st.session_state.simulation_runs = 0
if "simulation_last_run" not in st.session_state:
    st.session_state.simulation_last_run = "Not run yet"



# BASIC HELPERS

def render_html(html: str) -> None:
    st.markdown(html.strip(), unsafe_allow_html=True)


def df(data: Dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    value = data.get(key, pd.DataFrame())
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def has_cols(frame: pd.DataFrame, cols: List[str]) -> bool:
    return not frame.empty and all(col in frame.columns for col in cols)


def money(value: float) -> str:
    return f"P {float(value):,.0f}"


def pct(value: float) -> str:
    return f"{float(value):.1f}%"


def format_forecast_value(metric: str, value: float) -> str:
    metric_lower = str(metric).lower()
    value = float(value) if pd.notna(value) else 0.0

    if "roas" in metric_lower:
        return f"{value:.2f}x"
    if (
        "rate" in metric_lower
        or "roi" in metric_lower
        or "attendance" in metric_lower
        or "capacity" in metric_lower
        or "performance" in metric_lower
    ):
        return pct(value)
    if "risk" in metric_lower:
        return f"{value:.0f}"
    if (
        "cpa" in metric_lower
        or "deal" in metric_lower
        or "revenue" in metric_lower
        or "profit" in metric_lower
        or "pipeline" in metric_lower
    ):
        return money(value)
    return f"{value:,.0f}"


def format_forecast_delta(metric: str, value: float) -> str:
    metric_lower = str(metric).lower()
    value = float(value) if pd.notna(value) else 0.0
    sign = "+" if value >= 0 else ""

    if "roas" in metric_lower:
        return f"{sign}{value:.2f}x"
    if (
        "rate" in metric_lower
        or "roi" in metric_lower
        or "attendance" in metric_lower
        or "capacity" in metric_lower
        or "performance" in metric_lower
    ):
        return f"{sign}{value:.1f}%"
    if "risk" in metric_lower:
        return f"{sign}{value:.0f}"
    if (
        "cpa" in metric_lower
        or "deal" in metric_lower
        or "revenue" in metric_lower
        or "profit" in metric_lower
        or "pipeline" in metric_lower
    ):
        return f"{sign}{money(value)}"
    return f"{sign}{value:,.0f}"



def safe_sum(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum())


def safe_mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0


def forecast_metric_options(department: str) -> List[str]:
    dept = (department or "Overview").strip().lower()
    if dept == "sales":
        return ["Pipeline Revenue", "Win Rate", "Average Deal Size", "Sales Cycle"]
    if dept == "marketing and advertising":
        return ["Leads", "Conversions", "ROAS", "CPA"]
    if dept == "hr":
        return ["Performance Score", "Operational Capacity", "Risk Staff", "Productivity Score"]
    return ["Revenue", "Profit", "ROI", "Conversions"]


def department_weight_profile(department: str, metric: str) -> Dict[str, float]:
    dept = (department or "Overview").strip().lower()
    metric_name = (metric or "Revenue").strip().lower()

    profile = {"demand": 0.45, "price": 0.25, "spend": 0.20, "capacity": 0.10}

    if dept == "sales":
        profile = {"demand": 0.42, "price": 0.28, "spend": 0.12, "capacity": 0.18}
    elif dept == "marketing and advertising":
        profile = {"demand": 0.30, "price": 0.08, "spend": 0.47, "capacity": 0.15}
    elif dept == "hr":
        profile = {"demand": 0.08, "price": 0.04, "spend": 0.20, "capacity": 0.68}

    if "capacity" in metric_name or "productivity" in metric_name or "risk" in metric_name:
        profile["capacity"] += 0.12
        profile["demand"] -= 0.05
        profile["spend"] -= 0.04
        profile["price"] -= 0.03
    elif "roas" in metric_name or "cpa" in metric_name or "leads" in metric_name or "conversions" in metric_name:
        profile["spend"] += 0.10
        profile["demand"] += 0.05
        profile["price"] -= 0.05
        profile["capacity"] -= 0.10
    elif "win rate" in metric_name or "deal" in metric_name or "cycle" in metric_name:
        profile["demand"] += 0.08
        profile["price"] += 0.04
        profile["spend"] -= 0.04
        profile["capacity"] -= 0.08

    profile = {k: max(v, 0.01) for k, v in profile.items()}
    total = sum(profile.values())
    return {k: v / total for k, v in profile.items()}


def current_simulation_params() -> Dict[str, float | str]:
    department = st.session_state.get("forecast_department", "All")
    metric = st.session_state.get("forecast_metric", forecast_metric_options(department)[0])
    weights = department_weight_profile(department, metric)
    params = {
        "view": st.session_state.get("forecast_view", "Enterprise"),
        "department": department,
        "metric": metric,
        "hr_department": st.session_state.get("forecast_hr_department", "All"),
        "horizon": int(st.session_state.get("forecast_horizon", 14)),
        "demand_shift": float(st.session_state.get("sim_demand_shift", 0)),
        "price_shift": float(st.session_state.get("sim_price_shift", 0)),
        "spend_shift": float(st.session_state.get("sim_spend_shift", 0)),
        "capacity_shift": float(st.session_state.get("sim_capacity_shift", 0)),
        "weights": weights,
    }
    params["net_lift"] = (
        params["demand_shift"] * weights["demand"]
        + params["price_shift"] * weights["price"]
        + params["spend_shift"] * weights["spend"]
        + params["capacity_shift"] * weights["capacity"]
    ) / 100.0
    return params


def first_date_col(frame: pd.DataFrame) -> Optional[str]:
    for col in ["date", "period", "timestamp", "created_at", "month"]:
        if col in frame.columns:
            return col
    return None


def empty_visual(message: str) -> None:
    st.info(message)


def plot(fig: go.Figure, height: int = 250) -> None:
    existing_legend = fig.layout.legend.to_plotly_json() if fig.layout.legend else {}
    existing_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    has_custom_legend = bool(existing_legend) and any(v is not None for v in existing_legend.values())
    has_custom_margin = bool(existing_margin) and any(v is not None for v in existing_margin.values())

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,14,26,.30)",
        font=dict(color="#F2FBFF", family="Inter, Segoe UI, Arial", size=9),
        title=dict(font=dict(size=13, color="#F2FBFF"), x=0.02, xanchor="left", y=.98),
        margin=existing_margin if has_custom_margin else dict(l=34, r=16, t=54, b=32),
        colorway=["#34A8FF", "#65DCFF", "#37DC8B", "#FFB84D", "#FF5C7A", "#A78BFA"],
    )
    if has_custom_legend:
        fig.update_layout(legend=existing_legend)
    else:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.10,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=8),
            )
        )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.055)", zeroline=False, tickfont=dict(size=8), automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.055)", zeroline=False, tickfont=dict(size=8), automargin=True)
    st.plotly_chart(fig, width="stretch")


def top_bottom_region(data: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    sales = df(data, "sales")
    if has_cols(sales, ["region", "revenue"]):
        perf = sales.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        if not perf.empty:
            return str(perf.iloc[0]["region"]), str(perf.iloc[-1]["region"])
    return "leading region", "weaker region"


def top_bottom_service(data: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    sales = df(data, "sales")
    if has_cols(sales, ["service", "revenue"]):
        perf = sales.groupby("service", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        if not perf.empty:
            return str(perf.iloc[0]["service"]), str(perf.iloc[-1]["service"])
    return "leading service", "weaker service"



# DATA MODEL HELPERS
def build_drilldown_views(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Keeps drill-down logic alive for joined data analysis.
    We do not render drill-down tables because the UI direction is visual-first.
    """
    try:
        joined = join_data(data)
        if isinstance(joined, dict):
            return joined
    except Exception:
        pass
    return {}


def regional_scorecard(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    sales = df(data, "sales")
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")

    if sales.empty or "region" not in sales.columns:
        return pd.DataFrame()

    score = sales.groupby("region", as_index=False).agg(
        revenue=("revenue", "sum") if "revenue" in sales.columns else ("region", "count"),
        transactions=("region", "count"),
    )

    if has_cols(marketing, ["region", "conversions"]):
        score = score.merge(
            marketing.groupby("region", as_index=False)["conversions"].sum(),
            on="region",
            how="left",
        )
    else:
        score["conversions"] = 0

    if has_cols(advertising, ["region", "roas"]):
        score = score.merge(
            advertising.groupby("region", as_index=False)["roas"].mean(),
            on="region",
            how="left",
        )
    else:
        score["roas"] = 0

    if has_cols(hr, ["region", "performance_score"]):
        score = score.merge(
            hr.groupby("region", as_index=False)["performance_score"].mean(),
            on="region",
            how="left",
        )
    else:
        score["performance_score"] = 0

    score = score.fillna(0)
    score["score"] = (
        score["revenue"].rank(pct=True) * 35
        + score["conversions"].rank(pct=True) * 25
        + score["roas"].rank(pct=True) * 25
        + score["performance_score"].rank(pct=True) * 15
    )
    score["score"] = score["score"].round(1)

    return score.sort_values("score", ascending=False)

# AI and SIGNAL ENGINE
def make_signals(data: Dict[str, pd.DataFrame]) -> List[Dict[str, str]]:
    k = executive_kpis(data)
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")
    top_region, weak_region = top_bottom_region(data)

    signals: List[Dict[str, str]] = []
    roi = float(k.get("roi", 0))

    if roi < 10:
        signals.append({"level": "danger", "arrow": "↓", "title": "ROI below threshold", "text": f"ROI {roi:.1f}%. Margin recovery needed."})
    elif roi < 25:
        signals.append({"level": "warning", "arrow": "→", "title": "ROI watch", "text": f"ROI {roi:.1f}%. Scale selectively."})
    else:
        signals.append({"level": "success", "arrow": "↑", "title": "ROI healthy", "text": f"ROI {roi:.1f}%. Return quality strong."})

    if has_cols(advertising, ["roas", "cpa"]):
        roas = safe_mean(advertising, "roas")
        if roas < 1.5:
            signals.append({"level": "danger", "arrow": "↓", "title": "Ad risk", "text": f"ROAS {roas:.2f}x. Reallocate spend."})
        elif roas < 3:
            signals.append({"level": "warning", "arrow": "→", "title": "Ad watch", "text": f"ROAS {roas:.2f}x. Monitor fatigue."})
        else:
            signals.append({"level": "success", "arrow": "↑", "title": "Ad strength", "text": f"ROAS {roas:.2f}x. Scale winners."})

    if has_cols(marketing, ["leads", "conversions"]):
        leads = safe_sum(marketing, "leads")
        conversions = safe_sum(marketing, "conversions")
        conv = conversions / leads * 100 if leads else 0
        signals.append({
            "level": "warning" if conv < 12 else "success",
            "arrow": "→" if conv < 12 else "↑",
            "title": "Demand quality",
            "text": f"Lead conversion {conv:.1f}%."
        })

    if has_cols(hr, ["attendance_rate", "performance_score"]):
        attendance = safe_mean(hr, "attendance_rate")
        performance = safe_mean(hr, "performance_score")
        if attendance < 85:
            signals.append({"level": "danger", "arrow": "↓", "title": "Capacity risk", "text": f"Attendance {attendance:.1f}%."})
        else:
            signals.append({"level": "success", "arrow": "↑", "title": "People stable", "text": f"Perf {performance:.2f}/5."})

    signals.append({"level": "info", "arrow": "→", "title": "Regional focus", "text": f"{top_region} leads; review {weak_region}."})

    return signals[:4]


def make_ai(data: Dict[str, pd.DataFrame], context: str) -> Dict[str, object]:
    k = executive_kpis(data)
    base = insight_text(data)
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")

    revenue = float(k.get("revenue", 0))
    profit = float(k.get("profit", 0))
    roi = float(k.get("roi", 0))
    conversions = float(k.get("conversions", 0))

    score = 0
    score += 25 if revenue > 0 else 0
    score += 25 if profit > 0 else -10
    score += 25 if roi >= 25 else 14 if roi >= 10 else -5
    score += 25 if conversions > 0 else 0
    score = max(0, min(100, score))

    if score >= 76:
        posture, level = "Scale with control", "success"
    elif score >= 50:
        posture, level = "Fix weak spots first", "warning"
    else:
        posture, level = "Stabilise before growth", "danger"

    top_region, weak_region = top_bottom_region(data)
    top_service, weak_service = top_bottom_service(data)

    drivers = [
        f"{top_service} leads revenue.",
        f"{top_region} is the strongest region.",
    ]

    risks: List[str] = []

    if has_cols(marketing, ["channel", "conversions"]):
        top_channel = marketing.groupby("channel", as_index=False)["conversions"].sum().sort_values("conversions", ascending=False).head(1)
        if not top_channel.empty:
            drivers.append(f"{top_channel.iloc[0]['channel']} leads conversions.")

    if has_cols(advertising, ["platform", "roas"]):
        weak_platform = advertising.groupby("platform", as_index=False)["roas"].mean().sort_values("roas").head(1)
        if not weak_platform.empty:
            risks.append(f"{weak_platform.iloc[0]['platform']} has weakest ROAS.")

    if has_cols(hr, ["department", "performance_score"]):
        weak_department = hr.groupby("department", as_index=False)["performance_score"].mean().sort_values("performance_score").head(1)
        if not weak_department.empty:
            risks.append(f"{weak_department.iloc[0]['department']} needs support.")

    actions = [
        f"Scale {top_region} and {top_service}.",
        f"Diagnose {weak_region} and {weak_service}.",
        "Protect ROI before expanding spend.",
    ]

    return {
        "score": score,
        "posture": posture,
        "level": level,
        "summary": base.get("insight", "Live performance is moving across revenue, demand, people and regions."),
        "cause": base.get("cause", "Movement is shaped by region, channel, service mix and workforce capacity."),
        "risk": base.get("risk", "Risk sits in inefficient spend, weak regions and capacity pressure."),
        "recommendation": base.get("recommendation", actions[0]),
        "drivers": drivers[:3],
        "risks": risks[:2] or ["Monitor ROI and capacity pressure."],
        "actions": actions[:3],
    }

# UI BLOCKS
def render_settings_popover() -> None:
    if hasattr(st, "popover"):
        with st.popover("⚙"):
            st.markdown("**Live dashboard controls**")
            st.toggle("Live mode", key="live_mode")
            st.selectbox(
                "Refresh",
                [10, 20, 30, 60],
                index=[10, 20, 30, 60].index(st.session_state.refresh_speed),
                format_func=lambda x: f"{x}s",
                key="refresh_speed",
            )
            st.slider("Rows/tick", 1, 25, key="rows_per_tick")
            st.caption(f"Updated {st.session_state.last_update}")
    else:
        with st.expander("⚙"):
            st.toggle("Live mode", key="live_mode")
            st.selectbox(
                "Refresh",
                [10, 20, 30, 60],
                index=[10, 20, 30, 60].index(st.session_state.refresh_speed),
                format_func=lambda x: f"{x}s",
                key="refresh_speed",
            )
            st.slider("Rows/tick", 1, 25, key="rows_per_tick")
    render_html("<div class='settings-caption'>Settings</div>")


def render_logs_popover() -> None:
    if hasattr(st, "popover"):
        with st.popover("Logs"):
            if st.session_state.live_log:
                for event in st.session_state.live_log[:8]:
                    st.caption(event)
            else:
                st.caption("No live ticks yet.")
    else:
        with st.expander("Logs"):
            if st.session_state.live_log:
                for event in st.session_state.live_log[:8]:
                    st.caption(event)
            else:
                st.caption("No live ticks yet.")


def _nav_slug(page_name: str) -> str:
    return page_name.lower().replace(" ", "_").replace("/", "_").replace("&", "and")


def _nav_label(page_name: str) -> str:
    labels = {
        "Overview": "Overview",
        "Sales": "Sales",
        "Marketing and Advertising": "Marketing\nand Advertising",
        "HR": "HR",
        "Forecast and Simulation Page": "Forecast\nSimulation",
        "Report": "Report",
    }
    return labels.get(page_name, page_name)


def render_top_bar() -> str:
    pages = [
        "Overview",
        "Sales",
        "Marketing and Advertising",
        "HR",
        "Forecast and Simulation Page",
        "Report",
    ]

    if st.session_state.get("active_page") not in pages:
        st.session_state.active_page = "Overview"

    with st.container(key="top_bar"):
        title_col, nav_col, settings_col, status_col = st.columns([0.78, 4.05, 0.52, 0.72])

        with title_col:
            with st.container(key="title_cell"):
                render_html("<div class='title-main'>CYBER NOVA BI</div><div class='title-sub'>Executive Intelligence Platform</div>")

        with nav_col:
            with st.container(key="navigation_cell"):
                nav_cols = st.columns([1, 1, 1.38, .68, 1.55, .80], gap="small")
                for nav_col_item, page_name in zip(nav_cols, pages):
                    is_active = st.session_state.active_page == page_name
                    nav_container_key = f"nav_active_{_nav_slug(page_name)}" if is_active else f"nav_idle_{_nav_slug(page_name)}"
                    with nav_col_item:
                        with st.container(key=nav_container_key):
                            if st.button(_nav_label(page_name), key=f"btn_{_nav_slug(page_name)}", use_container_width=True):
                                st.session_state.active_page = page_name
                                st.rerun()

        with settings_col:
            with st.container(key="settings_cell"):
                render_settings_popover()

        with status_col:
            with st.container(key="status_cell"):
                render_html(
                    f"<div class='status-stack'><div class='status-title'>Deploy</div><div class='live-cell'><span class='live-dot'></span>{'LIVE' if st.session_state.live_mode else 'PAUSED'}</div></div>"
                )

    return st.session_state.active_page


def render_forecast_control_bar(departments: List[str], hr_departments: List[str] | None = None) -> None:
    hr_departments = hr_departments or ["All"]

    if st.session_state.get("forecast_department") not in departments:
        st.session_state.forecast_department = departments[0]

    metric_options = forecast_metric_options(st.session_state.get("forecast_department", "Overview"))
    if st.session_state.get("forecast_metric") not in metric_options:
        st.session_state.forecast_metric = metric_options[0]

    with st.container(key="forecast_control_bar"):
        render_html("<div class='forecast-control-caption'>Forecast and Simulation Controls</div>")

        row1 = st.columns([0.88, 1.05, 1.15, 1.12, 0.78], gap="small")

        with row1[0]:
            st.selectbox("Function", ["Forecast", "Simulation"], key="forecast_mode")

        with row1[1]:
            st.selectbox("Forecast Area", departments, key="forecast_department")

        metric_options = forecast_metric_options(st.session_state.get("forecast_department", "Overview"))
        if st.session_state.get("forecast_metric") not in metric_options:
            st.session_state.forecast_metric = metric_options[0]

        with row1[2]:
            st.selectbox("Metric", metric_options, key="forecast_metric")

        with row1[3]:
            if st.session_state.get("forecast_department") == "HR":
                if st.session_state.get("forecast_hr_department") not in hr_departments:
                    st.session_state.forecast_hr_department = hr_departments[0]
                st.selectbox("HR department filter", hr_departments, key="forecast_hr_department")
            else:
                st.selectbox("Scope", ["Organisation", "Department"], key="forecast_view")

        with row1[4]:
            st.select_slider("Horizon", options=[3, 6, 9, 12], key="forecast_horizon")

        if st.session_state.forecast_mode == "Simulation":
            row2 = st.columns([1, 1, 1, 1, 0.85], gap="small")

            with row2[0]:
                st.slider("Demand shift (%)", -20, 30, key="sim_demand_shift")
            with row2[1]:
                st.slider("Pricing shift (%)", -15, 20, key="sim_price_shift")
            with row2[2]:
                st.slider("Marketing spend shift (%)", -20, 30, key="sim_spend_shift")
            with row2[3]:
                st.slider("Capacity shift (%)", -20, 20, key="sim_capacity_shift")
            with row2[4]:
                st.write("")
                if st.button("Run Simulation", use_container_width=True, key="run_forecast_sim"):
                    st.session_state.simulation_runs += 1
                    st.session_state.simulation_last_run = datetime.now().strftime("%H:%M:%S")
                    st.success("Simulation updated.")

            params = current_simulation_params()
            hr_filter = f" &nbsp;&nbsp; <b>HR filter:</b> {params['hr_department']}" if params["department"] == "HR" else ""
            render_html(
                "<div class='forecast-summary-strip'>"
                f"<b>Function:</b> Simulation &nbsp;&nbsp; "
                f"<b>Model:</b> Linear Regression baseline + what-if adjustment &nbsp;&nbsp; "
                f"<b>Area:</b> {params['department']} &nbsp;&nbsp; "
                f"<b>Metric:</b> {params['metric']}"
                f"{hr_filter} &nbsp;&nbsp; "
                f"<b>Projected lift:</b> {params['net_lift']*100:.1f}% &nbsp;&nbsp; "
                f"<b>Runs:</b> {st.session_state.simulation_runs} (last {st.session_state.simulation_last_run})"
                "</div>"
            )
        else:
            hr_filter = ""
            if st.session_state.get("forecast_department") == "HR":
                hr_filter = f" &nbsp;&nbsp; <b>HR filter:</b> {st.session_state.forecast_hr_department}"
            render_html(
                "<div class='forecast-summary-strip'>"
                f"<b>Function:</b> Forecast &nbsp;&nbsp; "
                f"<b>Model:</b> Linear Regression &nbsp;&nbsp; "
                f"<b>Area:</b> {st.session_state.forecast_department} &nbsp;&nbsp; "
                f"<b>Metric:</b> {st.session_state.forecast_metric}"
                f"{hr_filter} &nbsp;&nbsp; "
                f"<b>Horizon:</b> {st.session_state.forecast_horizon} periods"
                "</div>"
            )


def render_filter_panel(countries, regions, segments, services, page):
    with st.container(key="filter_panel"):
        render_html("<div class='panel-heading center'>Filters</div>")
        country = st.selectbox("Country", countries, key="filter_country")
        region = st.selectbox("Region", regions, key="filter_region")
        period = st.selectbox("Time Period", ["Last 30 days", "Last 7 days", "Last 90 days", "Historic", "All"], key="filter_period")
        segment = st.selectbox("Customer Segment", segments, key="filter_segment")
        service = st.selectbox("Service Line", services, key="filter_service")
        log_lines = st.session_state.live_log[:4] if st.session_state.live_log else ["No live ticks yet."]
        log_html = "".join([f"<div class='filter-log-line'>{line}</div>" for line in log_lines])
        render_html(
            "<div class='filter-log-box'>"
            "<div class='filter-log-title'>Logs</div>"
            f"{log_html}"
            "</div>"
        )
    return country, region, period, segment, service


def forecast_kpi_cards(data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str, str, str]]:
    department = st.session_state.get("forecast_department", "Overview")
    metric = st.session_state.get("forecast_metric", "Revenue")
    periods = int(st.session_state.get("forecast_horizon", 6))
    hr_department = st.session_state.get("forecast_hr_department", "All")
    mode = st.session_state.get("forecast_mode", "Forecast")

    if mode == "Simulation":
        scenario = linear_regression_simulation(
            data=data,
            department=department,
            metric=metric,
            periods=periods,
            demand_shift=float(st.session_state.get("sim_demand_shift", 0)),
            price_shift=float(st.session_state.get("sim_price_shift", 0)),
            spend_shift=float(st.session_state.get("sim_spend_shift", 0)),
            capacity_shift=float(st.session_state.get("sim_capacity_shift", 0)),
            hr_department=hr_department,
        )

        forecast_area = department if department != "HR" else f"HR • {hr_department}"

        if scenario.empty:
            return [
                ("KPI 1", "Simulation Confidence", "0%", "no simulation data", "→"),
                ("KPI 2", f"Scenario {metric}", "0", "scenario output", "→"),
                ("KPI 3", f"{metric} Uplift", "0", "vs baseline", "→"),
                ("KPI 4", "Forecast Area", forecast_area, "selected view", "→"),
            ]

        baseline_final = float(pd.to_numeric(scenario["baseline"], errors="coerce").fillna(0).iloc[-1])
        scenario_final = float(pd.to_numeric(scenario["scenario"], errors="coerce").fillna(0).iloc[-1])
        uplift_final = float(pd.to_numeric(scenario["uplift"], errors="coerce").fillna(0).iloc[-1])
        risk_level = str(scenario["risk_level"].iloc[-1]) if "risk_level" in scenario.columns else "Moderate"

        scenario_change = ((scenario_final - baseline_final) / baseline_final * 100) if baseline_final else 0.0
        confidence = max(72, min(92, 80 + st.session_state.get("simulation_runs", 0)))

        return [
            ("KPI 1", "Simulation Confidence", f"{confidence}%", "scenario model", "↑"),
            ("KPI 2", f"Scenario {metric}", format_forecast_value(metric, scenario_final), "adjusted outcome", "↑" if scenario_change >= 0 else "↓"),
            ("KPI 3", f"{metric} Uplift", format_forecast_delta(metric, uplift_final), "vs baseline", "↑" if uplift_final >= 0 else "↓"),
            ("KPI 4", "Risk Level", risk_level, forecast_area, "→"),
        ]

    forecast = linear_regression_forecast(
        data=data,
        department=department,
        metric=metric,
        periods=periods,
        hr_department=hr_department,
    )

    actual = forecast[forecast["type"] == "Actual"] if not forecast.empty and "type" in forecast.columns else pd.DataFrame()
    future = forecast[forecast["type"] == "Forecast"] if not forecast.empty and "type" in forecast.columns else pd.DataFrame()

    last_actual = float(pd.to_numeric(actual["value"], errors="coerce").fillna(0).iloc[-1]) if not actual.empty else 0.0
    predicted = float(pd.to_numeric(future["value"], errors="coerce").fillna(0).iloc[-1]) if not future.empty else last_actual
    change = ((predicted - last_actual) / last_actual * 100) if last_actual else 0.0

    confidence = float(forecast["accuracy"].iloc[-1]) if not forecast.empty and "accuracy" in forecast.columns else max(55, min(95, 62 + len(actual) * 5))
    forecast_area = department if department != "HR" else f"HR • {hr_department}"

    return [
        ("KPI 1", "Forecast Confidence", f"{confidence:.1f}%", "linear regression", "↑"),
        ("KPI 2", f"Predicted {metric}", format_forecast_value(metric, predicted), "predicted output", "↑" if change >= 0 else "↓"),
        ("KPI 3", f"{metric} Change", pct(change), "vs last actual", "↑" if change >= 0 else "↓"),
        ("KPI 4", "Forecast Area", forecast_area, "selected view", "→"),
    ]

def render_kpi_row(data: Dict[str, pd.DataFrame], page: str) -> None:
    with st.container(key="kpi_row"):
        k = executive_kpis(data)

        if page == "Sales":
            sm = sales_metrics(data)
            cards = [
                ("KPI 1", "Pipeline", money(sm.get("pipeline_value", 0)), "active value", "↑"),
                ("KPI 2", "Win Rate", pct(sm.get("win_rate", 0)), "qualified deals", "→"),
                ("KPI 3", "Avg Deal", money(sm.get("avg_deal_size", 0)), "deal quality", "↑"),
                ("KPI 4", "Cycle", f"{float(sm.get('avg_cycle', 0)):.1f}d", "speed", "→"),
            ]

        elif page == "Marketing and Advertising":
            mm = marketing_metrics(data)
            am = advertising_metrics(data)
            cards = [
                ("KPI 1", "Leads", f"{float(mm.get('leads', 0)):,.0f}", "demand", "↑"),
                ("KPI 2", "Conversion", pct(mm.get("conversion_rate", 0)), "efficiency", "↑"),
                ("KPI 3", "ROAS", f"{float(am.get('roas', 0)):.2f}x", "return", "→"),
                ("KPI 4", "CPA", money(am.get("cpa", 0)), "cost", "↓"),
            ]

        elif page == "HR":
            hm = hr_metrics(data)
            cards = [
                ("KPI 1", "Performance", f"{float(hm.get('avg_performance', 0)):.2f}/5", "output", "↑"),
                ("KPI 2", "Attendance", pct(hm.get("avg_attendance", 0)), "capacity", "→"),
                ("KPI 3", "Training", f"{float(hm.get('training_hours', 0)):,.0f}", "hours", "↑"),
                ("KPI 4", "Risk Staff", f"{int(hm.get('high_risk', 0))}", "attention", "↓"),
            ]

        elif page == "Forecast and Simulation Page":
            cards = forecast_kpi_cards(data)

        else:
            cards = [
                ("KPI 1", "Total Revenue", money(k.get("revenue", 0)), "commercial base", "↑"),
                ("KPI 2", "Profit", money(k.get("profit", 0)), "after costs", "→"),
                ("KPI 3", "ROI", pct(k.get("roi", 0)), "return quality", "→"),
                ("KPI 4", "Conversions", f"{float(k.get('conversions', 0)):,.0f}", "outcomes", "↑"),
            ]

        cols = st.columns(4)

        for col, (kpi_id, label, value, note, arrow) in zip(cols, cards):
            arrow_class = "arrow-up" if arrow == "↑" else "arrow-down" if arrow == "↓" else "arrow-info"
            with col:
                render_html(
                    "<div class='kpi-card'>"
                    f"<div class='kpi-id'>{kpi_id}</div>"
                    f"<div class='kpi-label'>{label}</div>"
                    f"<div class='kpi-value'><span class='{arrow_class}'>{arrow}</span> {value}</div>"
                    f"<div class='kpi-note'>{note}</div>"
                    "</div>"
                )


def moving_banner(data: Dict[str, pd.DataFrame]) -> None:
    k = executive_kpis(data)
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")
    sales = df(data, "sales")
    top_region, weak_region = top_bottom_region(data)
    top_service, weak_service = top_bottom_service(data)

    extra_signals: List[Dict[str, str]] = [
        {"level": "info", "arrow": "→", "title": "Revenue pulse", "text": f"{money(k.get('revenue', 0))} live commercial base."},
        {"level": "info", "arrow": "→", "title": "Profit watch", "text": f"{money(k.get('profit', 0))} after costs."},
        {"level": "success", "arrow": "↑", "title": "Conversion volume", "text": f"{float(k.get('conversions', 0)):,.0f} outcomes tracked."},
        {"level": "info", "arrow": "→", "title": "Regional spread", "text": f"{top_region} leading; {weak_region} needs review."},
        {"level": "info", "arrow": "→", "title": "Service mix", "text": f"{top_service} leads; monitor {weak_service}."},
    ]

    if has_cols(marketing, ["leads", "conversions"]):
        leads = safe_sum(marketing, "leads")
        conversions = safe_sum(marketing, "conversions")
        conv_rate = conversions / leads * 100 if leads else 0
        extra_signals.append({"level": "success" if conv_rate >= 12 else "warning", "arrow": "↑" if conv_rate >= 12 else "→", "title": "Lead engine", "text": f"{leads:,.0f} leads at {conv_rate:.1f}% conversion."})

    if has_cols(advertising, ["roas"]):
        extra_signals.append({"level": "success", "arrow": "↑", "title": "ROAS signal", "text": f"Average ROAS {safe_mean(advertising, 'roas'):.2f}x across platforms."})

    if has_cols(hr, ["attendance_rate", "performance_score"]):
        extra_signals.append({"level": "success", "arrow": "↑", "title": "People capacity", "text": f"Attendance {safe_mean(hr, 'attendance_rate'):.1f}%; performance {safe_mean(hr, 'performance_score'):.2f}/5."})

    if has_cols(sales, ["pipeline_stage"]):
        stage = sales["pipeline_stage"].astype(str).value_counts().head(1)
        if not stage.empty:
            extra_signals.append({"level": "info", "arrow": "→", "title": "Pipeline stage", "text": f"{stage.index[0]} has the highest activity ({int(stage.iloc[0]):,} records)."})

    extra_signals.append({"level": "info", "arrow": "→", "title": "Live refresh", "text": f"Last update {st.session_state.get('last_update', 'pending')} • {st.session_state.get('rows_per_tick', 0)} rows per tick."})

    items = ""
    for signal in make_signals(data) + extra_signals:
        arrow_class = "arrow-up" if signal["level"] == "success" else "arrow-down" if signal["level"] == "danger" else "arrow-watch" if signal["level"] == "warning" else "arrow-info"
        items += (
            "<span class='ticker-item'>"
            f"<span class='{arrow_class}'>{signal['arrow']}</span> "
            f"<span class='ticker-label'>{signal['title']}:</span> "
            f"{signal['text']}"
            "</span>"
        )

    render_html(
        "<div class='ticker'>"
        f"<div class='ticker-track'>{items}{items}</div>"
        "</div>"
    )


def assistant_panel(data: Dict[str, pd.DataFrame], context: str) -> None:
    ai = make_ai(data, context)
    items = (
        [f"<span class='arrow-up'>↑</span> {x}" for x in ai["drivers"]]
        + [f"<span class='arrow-watch'>→</span> {x}" for x in ai["risks"]]
        + [f"<span class='arrow-info'>→</span> {x}" for x in ai["actions"]]
    )

    render_html(
        "<div class='insight-panel-inner'>"
        f"<div class='panel-heading'>Insight Panel • {ai['posture']}</div>"
        "<div class='story-card'>"
        "<div class='story-label'>Insight</div>"
        f"<div class='story-text'><span class='arrow-up'>↑</span> {ai['summary']}</div>"
        "</div>"
        "<div class='story-card'>"
        "<div class='story-label'>Cause</div>"
        f"<div class='story-text'><span class='arrow-info'>→</span> {ai['cause']}</div>"
        "</div>"
        f"<div class='story-card {ai['level']}'>"
        "<div class='story-label'>Decision Posture</div>"
        f"<div class='story-text'>{ai['posture']}</div>"
        "<div class='score-line'>"
        f"<div class='score-fill' style='width:{ai['score']}%;'></div>"
        "</div>"
        "</div>"
        "<div class='ai-list'>"
        + "".join([f"<div class='ai-item'>{item}</div>" for item in items[:5]])
        + "</div></div>"
    )

# VISUALS
def region_map(data: Dict[str, pd.DataFrame], height: int = 520) -> None:
    map_data = region_summary(data)

    if map_data.empty or not has_cols(map_data, ["latitude", "longitude"]):
        empty_visual("Global map needs latitude and longitude.")
        return

    fig = px.scatter_map(
        map_data,
        lat="latitude",
        lon="longitude",
        size="opportunity_score" if "opportunity_score" in map_data.columns else None,
        color="market_status" if "market_status" in map_data.columns else None,
        hover_name="region" if "region" in map_data.columns else None,
        hover_data=[
            col for col in
            ["country", "revenue", "traffic", "customers", "regional_roas"]
            if col in map_data.columns
        ],
        zoom=2.05,
        title="Global Performance Map",
        height=height,
    )
    fig.update_layout(
        map_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F2FBFF",
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(orientation="h", y=1.00, x=1, xanchor="right", font=dict(size=9)),
    )
    st.plotly_chart(fig, width="stretch")


def revenue_trend(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    date_col = first_date_col(sales)

    if sales.empty or "revenue" not in sales.columns or date_col is None:
        empty_visual("Revenue trend needs date and revenue.")
        return

    temp = sales.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])

    if temp.empty:
        empty_visual("Date values could not be parsed.")
        return

    temp["period_group"] = temp[date_col].dt.to_period("D").dt.to_timestamp()
    trend = temp.groupby("period_group", as_index=False)["revenue"].sum()

    fig = px.line(trend, x="period_group", y="revenue", title="Revenue Trend")
    fig.update_traces(line=dict(width=2.2), mode="lines+markers", marker=dict(size=4))
    plot(fig, height=height)


def regional_score_visual(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    score = regional_scorecard(data)
    if score.empty:
        empty_visual("Regional scorecard needs regional data.")
        return

    fig = px.bar(
        score.head(8),
        x="score",
        y="region",
        orientation="h",
        color="score",
        title="Regional Performance Scorecard",
    )
    fig.update_layout(margin=dict(l=92, r=10, t=36, b=22))
    fig.update_yaxes(tickfont=dict(size=8), automargin=True)
    plot(fig, height=height)


def channel_mix(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")
    if marketing.empty or "channel" not in marketing.columns:
        empty_visual("Channel mix needs channel data.")
        return

    if "conversions" in marketing.columns:
        grouped = marketing.groupby("channel", as_index=False)["conversions"].sum().sort_values("conversions", ascending=False)
        value_col = "conversions"
    else:
        grouped = marketing["channel"].value_counts().reset_index()
        grouped.columns = ["channel", "count"]
        value_col = "count"

    fig = px.pie(grouped, names="channel", values=value_col, hole=.58, title="Channel Mix")
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        marker=dict(line=dict(color="#05111f", width=1)),
        hovertemplate="%{label}<br>%{value:,.0f}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=8, r=92, t=56, b=18),
        legend=dict(orientation="v", x=1.04, y=.95, xanchor="left", yanchor="top", font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
    )
    plot(fig, height=height)


def service_mix(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if not has_cols(sales, ["service", "revenue"]):
        empty_visual("Service mix needs service and revenue.")
        return
    grouped = sales.groupby("service", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    fig = px.bar(grouped, x="service", y="revenue", title="Service Revenue Mix")
    plot(fig, height=height)


def performance_opportunity_matrix(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    score = regional_scorecard(data)

    if score.empty or "revenue" not in score.columns or "score" not in score.columns:
        empty_visual("Strategic opportunity matrix needs regional scorecard data.")
        return

    score = score.copy()

    size_col = "conversions" if "conversions" in score.columns else "transactions" if "transactions" in score.columns else None
    if size_col is None:
        score["opportunity_size"] = 20
        size_col = "opportunity_size"

    revenue_mid = score["revenue"].median()
    score_mid = score["score"].median()

    def classify(row):
        if row["revenue"] >= revenue_mid and row["score"] >= score_mid:
            return "Strategic Leaders"
        elif row["revenue"] < revenue_mid and row["score"] >= score_mid:
            return "Growth Opportunities"
        elif row["revenue"] >= revenue_mid and row["score"] < score_mid:
            return "Stabilise & Optimise"
        else:
            return "Low Priority"

    score["strategic_zone"] = score.apply(classify, axis=1)

    fig = px.scatter(
        score,
        x="revenue",
        y="score",
        size=size_col,
        color="strategic_zone",
        text="region",
        hover_name="region",
        title="Strategic Opportunity Quadrant",
        labels={
            "revenue": "Revenue Strength",
            "score": "Opportunity / Performance Score",
            "strategic_zone": "Strategic Zone",
        },
        hover_data={
            "revenue": ":,.0f",
            "score": ":.1f",
            size_col: ":,.0f",
            "strategic_zone": True,
        },
    )

    fig.add_vline(
        x=revenue_mid,
        line_dash="dash",
        line_color="rgba(101,220,255,.55)",
    )

    fig.add_hline(
        y=score_mid,
        line_dash="dash",
        line_color="rgba(101,220,255,.55)",
    )

    max_revenue = score["revenue"].max()
    min_revenue = score["revenue"].min()
    max_score = score["score"].max()
    min_score = score["score"].min()

    x_left = min_revenue + (revenue_mid - min_revenue) * 0.45
    x_right = revenue_mid + (max_revenue - revenue_mid) * 0.55
    y_bottom = min_score + (score_mid - min_score) * 0.45
    y_top = score_mid + (max_score - score_mid) * 0.55

    fig.add_annotation(
        x=x_right,
        y=y_top,
        text="Strategic Leaders<br><span style='font-size:10px'>Protect & scale</span>",
        showarrow=False,
        font=dict(size=11),
        opacity=0.85,
    )

    fig.add_annotation(
        x=x_left,
        y=y_top,
        text="Growth Opportunities<br><span style='font-size:10px'>Invest & expand</span>",
        showarrow=False,
        font=dict(size=11),
        opacity=0.85,
    )

    fig.add_annotation(
        x=x_right,
        y=y_bottom,
        text="Stabilise & Optimise<br><span style='font-size:10px'>Improve efficiency</span>",
        showarrow=False,
        font=dict(size=11),
        opacity=0.85,
    )

    fig.add_annotation(
        x=x_left,
        y=y_bottom,
        text="Low Priority<br><span style='font-size:10px'>Monitor</span>",
        showarrow=False,
        font=dict(size=11),
        opacity=0.85,
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(
            sizemin=8,
            line=dict(width=1, color="rgba(255,255,255,.45)")
        )
    )

    fig.update_layout(
        margin=dict(l=46, r=10, t=42, b=34),
        legend_title_text="Strategic Zone",
        xaxis_title="Revenue Strength",
        yaxis_title="Opportunity / Performance Score",
    )

    plot(fig, height=height)

def regional_efficiency(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if not has_cols(sales, ["region", "revenue"]):
        empty_visual("Regional efficiency needs region and revenue.")
        return

    grouped = sales.groupby("region", as_index=False).agg(
        revenue=("revenue", "sum"),
        transactions=("revenue", "count"),
    )
    grouped["avg_revenue"] = grouped["revenue"] / grouped["transactions"].replace(0, pd.NA)

    fig = px.scatter(
        grouped,
        x="transactions",
        y="avg_revenue",
        size="revenue",
        color="region",
        title="Regional Opportunity Efficiency",
        hover_data=["revenue"],
    )
    plot(fig, height=height)


def sales_funnel(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if sales.empty or "pipeline_stage" not in sales.columns:
        empty_visual("Sales funnel needs pipeline_stage.")
        return

    stages = sales["pipeline_stage"].value_counts().reset_index()
    stages.columns = ["stage", "count"]
    fig = px.funnel(stages, x="count", y="stage", title="Sales Conversion Funnel")
    plot(fig, height=height)


def sales_pipeline_heatmap(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if not has_cols(sales, ["pipeline_stage", "revenue"]):
        empty_visual("Pipeline stage analysis needs pipeline_stage and revenue.")
        return

    stage_order = ["Qualified", "Proposal", "Negotiation", "Lead", "Closed"]
    grouped = sales.groupby("pipeline_stage", as_index=False)["revenue"].sum()
    grouped["pipeline_stage"] = pd.Categorical(grouped["pipeline_stage"], categories=stage_order, ordered=True)
    grouped = grouped.sort_values("pipeline_stage")
    fig = px.bar(grouped, x="revenue", y="pipeline_stage", orientation="h", title="Pipeline Opportunity by Stage", labels={"revenue":"Deal Value", "pipeline_stage":"Pipeline Stage"})
    plot(fig, height=height)


def deal_quality(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if sales.empty or "region" not in sales.columns:
        empty_visual("Regional sales quality needs regional sales data.")
        return

    if "cycle_days" in sales.columns:
        grouped = sales.groupby("region", as_index=False)["cycle_days"].mean().sort_values("cycle_days")
        fig = px.bar(grouped, x="region", y="cycle_days", title="Average Sales Cycle by Region", labels={"region":"Region", "cycle_days":"Avg Cycle Days"})
        fig.add_hline(y=float(grouped["cycle_days"].mean()), line_dash="dash", line_color="rgba(255,184,77,.85)")
    else:
        metric_col = "deal_size" if "deal_size" in sales.columns else "revenue"
        grouped = sales.groupby("region", as_index=False)[metric_col].mean().sort_values(metric_col, ascending=False)
        fig = px.bar(grouped, x="region", y=metric_col, title="Average Deal Size by Region")
    plot(fig, height=height)


def salesperson_performance(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if not has_cols(sales, ["salesperson", "revenue"]):
        empty_visual("Salesperson chart needs salesperson and revenue.")
        return

    grouped = sales.groupby("salesperson", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(7)
    fig = px.bar(grouped, x="salesperson", y="revenue", title="Salesperson Contribution")
    plot(fig, height=height)


def marketing_funnel(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")
    needed = ["impressions", "clicks", "leads", "conversions"]

    if marketing.empty or not all(col in marketing.columns for col in needed):
        empty_visual("Marketing funnel needs impressions, clicks, leads and conversions.")
        return

    funnel = pd.DataFrame({
        "stage": ["Impressions", "Clicks", "Leads", "Conversions"],
        "value": [safe_sum(marketing, col) for col in needed],
    })
    fig = px.funnel(funnel, x="value", y="stage", title="Demand Funnel")
    plot(fig, height=height)


def ad_efficiency(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    advertising = df(data, "advertising")
    if not has_cols(advertising, ["platform", "roas"]):
        empty_visual("Ad efficiency needs platform and ROAS.")
        return

    grouped = advertising.groupby("platform", as_index=False)["roas"].mean().sort_values("roas", ascending=False)
    fig = px.bar(grouped, x="platform", y="roas", text_auto=".2f", title="ROAS by Platform", labels={"platform":"Platform", "roas":"ROAS"})
    fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,184,77,.85)", annotation_text="watchline")
    plot(fig, height=height)


def marketing_channel_performance(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")
    if not has_cols(marketing, ["channel", "conversions"]):
        empty_visual("Channel performance needs channel and conversions.")
        return

    grouped = marketing.groupby("channel", as_index=False)["conversions"].sum().sort_values("conversions", ascending=False)
    fig = px.bar(grouped, x="channel", y="conversions", title="Conversions by Channel", labels={"channel":"Channel", "conversions":"Conversions"})
    plot(fig, height=height)


def campaign_heatmap(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")
    if not has_cols(marketing, ["channel", "region", "conversions"]):
        empty_visual("Activity density view needs channel, region and conversions.")
        return

    density = marketing.pivot_table(
        index="channel",
        columns="region",
        values="conversions",
        aggfunc="sum",
        fill_value=0,
    )
    fig = go.Figure(data=go.Contour(
        z=density.values,
        x=list(density.columns),
        y=list(density.index),
        colorscale=[[0, "#0f2b44"], [0.35, "#2a7fcb"], [0.65, "#37DC8B"], [1, "#FFB84D"]],
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=9, color="#F2FBFF")),
        line=dict(width=1),
        hovertemplate="Region: %{x}<br>Channel: %{y}<br>Activity: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Marketing Activity Density & Response")
    plot(fig, height=height)


def hr_training(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if not has_cols(hr, ["training_hours", "performance_score"]):
        empty_visual("HR chart needs training_hours and performance_score.")
        return

    fig = px.scatter(
        hr,
        x="training_hours",
        y="performance_score",
        size="productivity_score" if "productivity_score" in hr.columns else None,
        color="department" if "department" in hr.columns else None,
        hover_name="name" if "name" in hr.columns else None,
        title="Training vs Performance",
    )
    fig.add_hline(y=3.0, line_dash="dash", line_color="rgba(255,184,77,.85)", annotation_text="watchline")
    plot(fig, height=height)


def department_performance(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if not has_cols(hr, ["department", "performance_score"]):
        empty_visual("Department chart needs department and performance_score.")
        return

    grouped = hr.groupby("department", as_index=False)["performance_score"].mean().sort_values("performance_score")
    fig = px.bar(grouped, x="performance_score", y="department", orientation="h", title="Department Performance")
    plot(fig, height=height)


def attendance_risk(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if not has_cols(hr, ["department", "attendance_rate"]):
        empty_visual("Attendance risk needs department and attendance_rate.")
        return

    grouped = hr.groupby("department", as_index=False)["attendance_rate"].mean().sort_values("attendance_rate")
    fig = px.bar(grouped, x="department", y="attendance_rate", title="Operational Capacity Risk")
    fig.add_hline(y=85, line_dash="dash", line_color="rgba(255,92,122,.85)", annotation_text="risk")
    plot(fig, height=height)


def sales_velocity_treemap(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if sales.empty or "revenue" not in sales.columns:
        empty_visual("Revenue velocity view needs sales revenue data.")
        return

    group_col = "region" if "region" in sales.columns else "salesperson" if "salesperson" in sales.columns else None
    if group_col is None:
        empty_visual("Revenue velocity view needs region or salesperson data.")
        return

    temp = sales.copy()
    temp["deal_size_metric"] = pd.to_numeric(temp["deal_size"], errors="coerce").fillna(0) if "deal_size" in temp.columns else pd.to_numeric(temp["revenue"], errors="coerce").fillna(0)
    temp["cycle_metric"] = pd.to_numeric(temp["cycle_days"], errors="coerce").replace(0, np.nan).fillna(30) if "cycle_days" in temp.columns else pd.Series(30, index=temp.index)
    if "pipeline_stage" in temp.columns:
        temp["win_flag"] = temp["pipeline_stage"].astype(str).str.lower().eq("closed").astype(int)
    else:
        temp["win_flag"] = 0

    grouped = temp.groupby(group_col, as_index=False).agg(
        revenue=("revenue", "sum"),
        avg_deal=("deal_size_metric", "mean"),
        avg_cycle=("cycle_metric", "mean"),
        win_rate=("win_flag", "mean"),
    ).sort_values("revenue", ascending=False).head(10)

    if grouped.empty:
        empty_visual("Not enough grouped sales data for revenue velocity.")
        return

    grouped["velocity_score"] = (grouped["win_rate"].fillna(0) * 100) / grouped["avg_cycle"].replace(0, np.nan).fillna(grouped["avg_cycle"].mean())
    fig = px.scatter(
        grouped,
        x="avg_cycle",
        y="avg_deal",
        size="revenue",
        color="velocity_score",
        hover_name=group_col,
        title="Revenue Velocity Quadrant",
        labels={
            "avg_cycle": "Average Cycle Days",
            "avg_deal": "Average Deal Size",
            "velocity_score": "Velocity Score",
            "revenue": "Revenue",
        },
    )
    fig.add_vline(x=float(grouped["avg_cycle"].mean()), line_dash="dash", line_color="rgba(101,220,255,.45)")
    fig.add_hline(y=float(grouped["avg_deal"].mean()), line_dash="dash", line_color="rgba(101,220,255,.45)")
    fig.update_layout(margin=dict(l=34, r=16, t=48, b=34))
    plot(fig, height=height)


def sales_team_driver_radar(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")

    group_col = (
        "region" if "region" in sales.columns
        else "salesperson" if "salesperson" in sales.columns
        else None
    )

    if sales.empty or group_col is None or "revenue" not in sales.columns:
        empty_visual("Sales driver view needs grouped sales performance data.")
        return

    temp = sales.copy()

    temp["deal_size_metric"] = temp["deal_size"] if "deal_size" in temp.columns else temp["revenue"]
    temp["cycle_metric"] = temp["cycle_days"] if "cycle_days" in temp.columns else 30
    temp["activity_metric"] = 1

    if "pipeline_stage" in temp.columns:
        temp["win_flag"] = temp["pipeline_stage"].astype(str).str.lower().eq("closed").astype(int)
    elif "win_loss_status" in temp.columns:
        temp["win_flag"] = temp["win_loss_status"].astype(str).str.lower().eq("won").astype(int)
    else:
        temp["win_flag"] = 0

    grouped = temp.groupby(group_col, as_index=False).agg(
        revenue=("revenue", "sum"),
        avg_deal=("deal_size_metric", "mean"),
        activity=("activity_metric", "sum"),
        cycle_days=("cycle_metric", "mean"),
        win_rate=("win_flag", "mean"),
    )

    if grouped.empty:
        empty_visual("Not enough grouped sales data for the sales driver view.")
        return

    # Convert cycle time into velocity score: shorter cycle = better
    grouped["velocity_score"] = 1 / grouped["cycle_days"].replace(0, np.nan)
    grouped["velocity_score"] = grouped["velocity_score"].fillna(0)

    score_columns = {
        "Revenue Strength": "revenue",
        "Deal Size": "avg_deal",
        "Lead Activity": "activity",
        "Win Rate": "win_rate",
        "Sales Velocity": "velocity_score",
    }

    for score_name, col in score_columns.items():
        max_value = grouped[col].max()
        if max_value and max_value > 0:
            grouped[score_name] = grouped[col] / max_value * 100
        else:
            grouped[score_name] = 0

    grouped["Overall Driver Score"] = grouped[list(score_columns.keys())].mean(axis=1)

    top_groups = grouped.sort_values("Overall Driver Score", ascending=False).head(5)

    fig = px.bar(
        top_groups.sort_values("Overall Driver Score"),
        x="Overall Driver Score",
        y=group_col,
        orientation="h",
        text=top_groups.sort_values("Overall Driver Score")["Overall Driver Score"].round(1),
        title="Sales Team Driver Scorecard",
        hover_data={
            "Revenue Strength": ":.1f",
            "Deal Size": ":.1f",
            "Lead Activity": ":.1f",
            "Win Rate": ":.1f",
            "Sales Velocity": ":.1f",
            "Overall Driver Score": ":.1f",
        },
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Overall Sales Driver Score",
        yaxis_title=group_col.title(),
        showlegend=False,
    )

    plot(fig, height=height)


def sales_deal_stage_funnel(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    if sales.empty or "pipeline_stage" not in sales.columns:
        empty_visual("Sales funnel needs pipeline_stage.")
        return

    order = ["Qualified", "Proposal", "Negotiation", "Lead", "Closed"]
    stages = sales["pipeline_stage"].value_counts().reindex(order).fillna(0).reset_index()
    stages.columns = ["stage", "count"]
    fig = px.funnel(stages, x="count", y="stage", title="Deal Stage Funnel")
    plot(fig, height=height)


def sales_driver_correlation_matrix(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")

    group_col = (
        "region" if "region" in sales.columns
        else "salesperson" if "salesperson" in sales.columns
        else None
    )

    if sales.empty or group_col is None or "revenue" not in sales.columns:
        empty_visual("Strategic revenue drivers need grouped sales data.")
        return

    temp = sales.copy()

    temp["revenue"] = pd.to_numeric(temp["revenue"], errors="coerce").fillna(0)

    if "deal_size" in temp.columns:
        temp["deal_size_metric"] = pd.to_numeric(temp["deal_size"], errors="coerce").fillna(0)
    else:
        temp["deal_size_metric"] = temp["revenue"]

    if "cycle_days" in temp.columns:
        temp["cycle_metric"] = pd.to_numeric(temp["cycle_days"], errors="coerce").replace(0, np.nan).fillna(30)
    else:
        temp["cycle_metric"] = 30

    if "pipeline_stage" in temp.columns:
        temp["win_flag"] = temp["pipeline_stage"].astype(str).str.lower().eq("closed").astype(int)
    elif "win_loss_status" in temp.columns:
        temp["win_flag"] = temp["win_loss_status"].astype(str).str.lower().eq("won").astype(int)
    else:
        temp["win_flag"] = 0

    temp["activity_metric"] = 1

    grouped = temp.groupby(group_col, as_index=False).agg(
        revenue=("revenue", "sum"),
        avg_deal_size=("deal_size_metric", "mean"),
        avg_cycle_days=("cycle_metric", "mean"),
        win_rate=("win_flag", "mean"),
        activity=("activity_metric", "sum"),
    )

    if grouped.empty:
        empty_visual("Not enough grouped sales driver data.")
        return

    grouped["sales_velocity"] = 1 / grouped["avg_cycle_days"].replace(0, np.nan)
    grouped["sales_velocity"] = grouped["sales_velocity"].fillna(0)

    # Normalize drivers into clear 0–100 scores
    driver_cols = {
        "Deal Size Score": "avg_deal_size",
        "Win Rate Score": "win_rate",
        "Activity Score": "activity",
        "Velocity Score": "sales_velocity",
    }

    for score_name, col in driver_cols.items():
        max_value = grouped[col].max()
        grouped[score_name] = (grouped[col] / max_value * 100) if max_value > 0 else 0

    grouped["Driver Strength"] = grouped[list(driver_cols.keys())].mean(axis=1)

    top_groups = grouped.sort_values("revenue", ascending=False).head(8)

    fig = px.scatter(
        top_groups,
        x="Driver Strength",
        y="revenue",
        size="avg_deal_size",
        color="win_rate",
        text=group_col,
        title="Strategic Revenue Drivers",
        labels={
            "Driver Strength": "Overall Driver Strength",
            "revenue": "Revenue",
            "avg_deal_size": "Average Deal Size",
            "win_rate": "Win Rate",
        },
        hover_data={
            group_col: True,
            "revenue": ":,.0f",
            "avg_deal_size": ":,.0f",
            "win_rate": ":.1%",
            "activity": ":,.0f",
            "avg_cycle_days": ":.1f",
            "Driver Strength": ":.1f",
        },
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(
            sizemin=8,
            line=dict(width=1)
        )
    )

    fig.update_layout(
        xaxis_title="Driver Strength Score",
        yaxis_title="Revenue",
        showlegend=True,
        margin=dict(l=22, r=22, t=48, b=10),
    )

    plot(fig, height=height)

def marketing_circular_funnel(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")
    if marketing.empty or not all(col in marketing.columns for col in ["impressions", "clicks", "leads", "conversions"]):
        empty_visual("Demand funnel needs impressions, clicks, leads and conversions.")
        return

    imp = max(safe_sum(marketing, "impressions"), 1)
    clk = max(safe_sum(marketing, "clicks"), 1)
    lds = max(safe_sum(marketing, "leads"), 1)
    cnv = max(safe_sum(marketing, "conversions"), 1)

    click_rate = clk / imp * 100
    lead_rate = lds / clk * 100 if clk else 0
    conversion_rate = cnv / lds * 100 if lds else 0
    total_conversion = cnv / imp * 100

    labels = ["Reach", "Clicks", "Leads", "Conversions"]
    display_values = [100, max(click_rate, 4), max(lead_rate, 4), max(conversion_rate, 4)]
    actual_values = [imp, clk, lds, cnv]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=display_values,
        hole=.58,
        sort=False,
        direction="clockwise",
        marker={"colors": ["#34A8FF", "#65DCFF", "#37DC8B", "#FFB84D"], "line": {"color": "rgba(255,255,255,.15)", "width": 1}},
        textinfo="label+percent",
        textposition="inside",
        customdata=actual_values,
        hovertemplate="%{label}<br>Actual: %{customdata:,.0f}<br>Stage weight: %{value:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="Circular Demand Funnel",
        showlegend=False,
        margin=dict(l=8, r=8, t=48, b=12),
    )
    fig.add_annotation(text=f"Demand<br><b>{total_conversion:.2f}%</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=17, color="#F2FBFF"), align="center")
    plot(fig, height=height)


def marketing_surface_map(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    advertising = df(data, "advertising")
    if advertising.empty:
        empty_visual("ROAS & risk view needs advertising data.")
        return

    temp = advertising.copy()
    if "platform" not in temp.columns:
        temp["platform"] = [f"Channel {i+1}" for i in range(len(temp))]
    if "spend" not in temp.columns:
        if all(c in temp.columns for c in ["cpa", "conversions"]):
            temp["spend"] = pd.to_numeric(temp["cpa"], errors="coerce").fillna(0) * pd.to_numeric(temp["conversions"], errors="coerce").fillna(0)
        elif all(c in temp.columns for c in ["revenue", "roas"]):
            temp["spend"] = pd.to_numeric(temp["revenue"], errors="coerce").fillna(0) / pd.to_numeric(temp["roas"], errors="coerce").replace(0, np.nan).fillna(1)
        else:
            temp["spend"] = 1

    def risk_color(roas: float) -> str:
        if roas >= 2.5:
            return "rgba(55,220,139,.86)"
        if roas >= 1.5:
            return "rgba(255,184,77,.86)"
        return "rgba(255,92,122,.86)"

    def channel_group(name: str) -> str:
        lower = str(name).lower()
        if any(k in lower for k in ["google", "search", "ppc"]):
            return "Digital"
        if any(k in lower for k in ["facebook", "instagram", "linkedin", "social"]):
            return "Social"
        if any(k in lower for k in ["email", "crm", "referral"]):
            return "Direct"
        return "Channel Mix"

    temp["channel_group"] = temp["platform"].apply(channel_group)
    grouped = temp.groupby(["channel_group", "platform"], as_index=False).agg(
        spend=("spend", "sum"),
        roas=("roas", "mean") if "roas" in temp.columns else ("spend", "mean")
    )
    grouped["color"] = grouped["roas"].apply(risk_color)
    parent_totals = grouped.groupby("channel_group", as_index=False)["spend"].sum()

    labels = ["Spend Mix"] + parent_totals["channel_group"].tolist() + grouped["platform"].tolist()
    parents = [""] + ["Spend Mix"] * len(parent_totals) + grouped["channel_group"].tolist()
    values = [grouped["spend"].sum()] + parent_totals["spend"].tolist() + grouped["spend"].tolist()
    colors = ["rgba(52,168,255,.92)"] + ["rgba(18,76,120,.86)"] * len(parent_totals) + grouped["color"].tolist()

    overall_roas = float(pd.to_numeric(grouped["roas"], errors="coerce").mean()) if not grouped.empty else 0.0
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,.12)", width=1)),
        insidetextorientation="radial",
        textinfo="label",
    ))
    fig.update_layout(title="Channel Spend & Risk Mix", showlegend=False, margin=dict(l=8, r=8, t=46, b=8))
    risk_label = "Low Risk" if overall_roas >= 2.5 else "Watch" if overall_roas >= 1.5 else "High Risk"
    fig.add_annotation(text=f"Spend<br><b>{risk_label}</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=15, color="#F2FBFF"))
    plot(fig, height=height)


def marketing_spend_waterfall(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    advertising = df(data, "advertising")
    if advertising.empty or "platform" not in advertising.columns:
        empty_visual("Spend waterfall needs platform-level advertising data.")
        return

    temp = advertising.copy()
    if "spend" not in temp.columns:
        if all(c in temp.columns for c in ["cpa", "conversions"]):
            temp["spend"] = pd.to_numeric(temp["cpa"], errors="coerce").fillna(0) * pd.to_numeric(temp["conversions"], errors="coerce").fillna(0)
        elif all(c in temp.columns for c in ["revenue", "roas"]):
            temp["spend"] = pd.to_numeric(temp["revenue"], errors="coerce").fillna(0) / pd.to_numeric(temp["roas"], errors="coerce").replace(0, np.nan).fillna(1)
        else:
            temp["spend"] = 0
    grouped = temp.groupby("platform", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
    measures = ["relative"] * len(grouped) + ["total"]
    x = grouped["platform"].tolist() + ["Total"]
    y = grouped["spend"].tolist() + [grouped["spend"].sum()]
    fig = go.Figure(go.Waterfall(x=x, measure=measures, y=y))
    fig.update_layout(title="Marketing Spend Waterfall")
    plot(fig, height=height)


def hr_parallel_coordinates(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if hr.empty or "department" not in hr.columns:
        empty_visual("Parallel coordinates need department-level HR data.")
        return

    temp = hr.copy()
    temp["engagement_proxy"] = pd.to_numeric(temp.get("engagement_score", temp.get("performance_score", 0)), errors="coerce").fillna(0)
    temp["burnout_proxy"] = 100 - pd.to_numeric(temp.get("attendance_rate", 85), errors="coerce").fillna(85)
    temp["comp_proxy"] = pd.to_numeric(temp.get("compensation", temp.get("productivity_score", 0)), errors="coerce").fillna(0)
    temp["turnover_proxy"] = 100 - pd.to_numeric(temp.get("attendance_rate", 85), errors="coerce").fillna(85)
    grouped = temp.groupby("department", as_index=False).agg(
        turnover=("turnover_proxy", "mean"),
        burnout=("burnout_proxy", "mean"),
        engagement=("engagement_proxy", "mean"),
        compensation=("comp_proxy", "mean"),
    )
    if grouped.empty:
        empty_visual("Not enough HR data for parallel coordinates.")
        return

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=grouped["engagement"], colorscale="Blues", showscale=True),
        dimensions=[
            dict(label="Turnover", values=grouped["turnover"]),
            dict(label="Burnout", values=grouped["burnout"]),
            dict(label="Engagement", values=grouped["engagement"]),
            dict(label="Compensation", values=grouped["compensation"]),
        ],
    ))
    fig.update_layout(title="Departmental Talent Drivers")
    plot(fig, height=height)


def hr_sentiment_gauge(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if hr.empty or "department" not in hr.columns:
        empty_visual("Sentiment gauge needs HR department data.")
        return

    temp = hr.copy()
    temp["sentiment_proxy"] = (
        pd.to_numeric(temp.get("attendance_rate", 85), errors="coerce").fillna(85) * 0.40
        + pd.to_numeric(temp.get("performance_score", 3), errors="coerce").fillna(3) * 18
        + pd.to_numeric(temp.get("productivity_score", 3), errors="coerce").fillna(3) * 12
    ).clip(0, 100)
    overall = float(temp["sentiment_proxy"].mean())

    if overall >= 75:
        posture = "Healthy"
    elif overall >= 55:
        posture = "Watch"
    else:
        posture = "At Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall,
        number={"suffix": "/100", "font": {"size": 26}},
        domain={"x": [0.04, 0.96], "y": [0.12, 0.88]},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#F2FBFF", "tickfont": {"size": 9}},
            "bar": {"color": "#6bdcff", "thickness": 0.26},
            "steps": [
                {"range": [0, 40], "color": "rgba(255,92,122,.35)"},
                {"range": [40, 70], "color": "rgba(255,184,77,.35)"},
                {"range": [70, 100], "color": "rgba(66,255,135,.28)"},
            ],
            "threshold": {"line": {"color": "#FFB84D", "width": 3}, "thickness": 0.75, "value": 70},
        },
    ))
    fig.add_annotation(
        x=0.5, y=0.02, xref="paper", yref="paper", showarrow=False,
        text=f"Culture pulse: <b>{posture}</b>",
        font=dict(size=11, color="#F2FBFF"),
    )
    fig.update_layout(title="Employee Sentiment Gauge", margin=dict(l=8, r=8, t=46, b=20), showlegend=False)
    plot(fig, height=height)


def hr_capacity_risk_line(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    hr = df(data, "hr")
    if hr.empty or "department" not in hr.columns:
        empty_visual("Capacity risk line needs department-level HR data.")
        return

    temp = hr.copy()
    temp["risk_index"] = 100 - pd.to_numeric(temp.get("attendance_rate", 85), errors="coerce").fillna(85)
    if "productivity_score" in temp.columns:
        temp["risk_index"] += (5 - pd.to_numeric(temp["productivity_score"], errors="coerce").fillna(3)) * 5
    grouped = temp.groupby("department", as_index=False)["risk_index"].mean().sort_values("risk_index", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["department"], y=grouped["risk_index"], mode="lines+markers", name="Risk index"))
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(255,92,122,.85)", annotation_text="safe threshold")
    fig.update_layout(title="Operational Capacity Risk Line")
    plot(fig, height=height)


# FORECASTING VISUALS
def forecast_base_series(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    sales = df(data, "sales")
    date_col = first_date_col(sales)

    if sales.empty or date_col is None or "revenue" not in sales.columns:
        return pd.DataFrame()

    temp = sales.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])

    if temp.empty:
        return pd.DataFrame()

    temp["period"] = temp[date_col].dt.to_period("D").dt.to_timestamp()
    return temp.groupby("period", as_index=False)["revenue"].sum().sort_values("period")


def forecast_confidence_chart(data: Dict[str, pd.DataFrame], height: int = 520, horizon: int = 10) -> None:
    params = current_simulation_params()
    actual = forecast_base_series(data)

    if actual.empty:
        empty_visual("Forecast needs historical date and revenue data.")
        return

    actual = actual.tail(45).copy()
    actual["pct_change"] = actual["revenue"].pct_change().replace([np.inf, -np.inf], np.nan)

    recent_change = actual["pct_change"].tail(10).dropna()
    trend = float(recent_change.mean()) if len(recent_change) else 0.02
    trend = max(min(trend + params['net_lift'] * 0.55, 0.22), -0.12)

    volatility = float(actual["pct_change"].dropna().std()) if len(actual["pct_change"].dropna()) else 0.08
    volatility = max(min(volatility, 0.22), 0.04)

    last_date = actual["period"].max()
    last_value = float(actual["revenue"].iloc[-1])

    future_rows = []
    for step in range(1, horizon + 1):
        forecast_value = last_value * ((1 + trend) ** step)
        scale = volatility * np.sqrt(step)
        future_rows.append({
            "period": last_date + timedelta(days=step),
            "forecast": forecast_value,
            "lower50": max(0, forecast_value * (1 - scale * 0.50)),
            "upper50": forecast_value * (1 + scale * 0.50),
            "lower80": max(0, forecast_value * (1 - scale * 0.90)),
            "upper80": forecast_value * (1 + scale * 0.90),
            "lower90": max(0, forecast_value * (1 - scale * 1.25)),
            "upper90": forecast_value * (1 + scale * 1.25),
        })

    future = pd.DataFrame(future_rows)

    fig = go.Figure()

    for upper, lower, color, name in [
        ("upper90", "lower90", "rgba(52,168,255,.12)", "90% confidence"),
        ("upper80", "lower80", "rgba(101,220,255,.16)", "80% confidence"),
        ("upper50", "lower50", "rgba(167,139,250,.20)", "50% confidence"),
    ]:
        fig.add_trace(go.Scatter(x=future["period"], y=future[upper], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future["period"], y=future[lower], mode="lines", fill="tonexty", fillcolor=color, line=dict(width=0), name=name))

    fig.add_trace(go.Scatter(
        x=actual["period"], y=actual["revenue"],
        mode="lines",
        line=dict(color="#F2FBFF", width=2.2),
        name="Historical actuals"
    ))

    fig.add_trace(go.Scatter(
        x=future["period"], y=future["forecast"],
        mode="lines+markers",
        line=dict(color="#65DCFF", width=3),
        marker=dict(size=5),
        name="Best estimate"
    ))

    fig.add_vline(x=last_date, line_dash="dash", line_color="rgba(255,255,255,.35)")
    fig.update_layout(title=f"{params['metric']} Forecast • {params['view']} / {params['department']}")
    plot(fig, height=height)


def forecast_driver_waterfall(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    params = current_simulation_params()
    k = executive_kpis(data)
    base = float(k.get("revenue", 0))

    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")

    conversion_boost = 0
    if has_cols(marketing, ["leads", "conversions"]):
        leads = safe_sum(marketing, "leads")
        conv = safe_sum(marketing, "conversions")
        conversion_rate = conv / leads if leads else 0
        conversion_boost = base * min(max(conversion_rate, 0), 0.35) * (1 + params['demand_shift']/100)

    roas_drag = 0
    if has_cols(advertising, ["roas"]):
        roas = safe_mean(advertising, "roas")
        roas_drag = base * (-0.06 if roas < 2 else 0.05 if roas > 4 else 0.01)
        roas_drag += base * (params['spend_shift'] / 100) * 0.18
        roas_drag += base * (params['price_shift'] / 100) * 0.10

    capacity_effect = 0
    if has_cols(hr, ["attendance_rate"]):
        attendance = safe_mean(hr, "attendance_rate")
        capacity_effect = base * (-0.05 if attendance < 85 else 0.03)
        capacity_effect += base * (params['capacity_shift'] / 100) * 0.16

    regional_effect = base * (0.08 + params['net_lift'] * 0.35)
    final = base + conversion_boost + roas_drag + capacity_effect + regional_effect

    fig = go.Figure(go.Waterfall(
        name="drivers",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Current", "Conversion", "Ad efficiency", "Capacity", "Regional expansion", "Forecast"],
        y=[base, conversion_boost, roas_drag, capacity_effect, regional_effect, final],
        connector={"line": {"color": "rgba(255,255,255,.25)"}},
        increasing={"marker": {"color": "#37DC8B"}},
        decreasing={"marker": {"color": "#FF5C7A"}},
        totals={"marker": {"color": "#65DCFF"}},
    ))
    fig.update_layout(title=f"Driver Impact on {params['metric']} • {params['department']}")
    plot(fig, height=height)


def predictive_risk_heatmap(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    sales = df(data, "sales")
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")

    if not has_cols(sales, ["region", "revenue"]):
        empty_visual("Risk heatmap needs regional revenue data.")
        return

    region_frame = sales.groupby("region", as_index=False)["revenue"].sum()

    if has_cols(marketing, ["region", "leads", "conversions"]):
        region_frame = region_frame.merge(marketing.groupby("region", as_index=False).agg(leads=("leads", "sum"), conversions=("conversions", "sum")), on="region", how="left")

    if has_cols(advertising, ["region", "roas", "cpa"]):
        region_frame = region_frame.merge(advertising.groupby("region", as_index=False).agg(roas=("roas", "mean"), cpa=("cpa", "mean")), on="region", how="left")

    if has_cols(hr, ["region", "attendance_rate", "performance_score"]):
        region_frame = region_frame.merge(hr.groupby("region", as_index=False).agg(attendance=("attendance_rate", "mean"), performance=("performance_score", "mean")), on="region", how="left")

    numeric = region_frame.drop(columns=["region"], errors="ignore").fillna(0)
    if numeric.shape[1] < 2:
        empty_visual("Risk heatmap needs at least two numeric drivers.")
        return

    corr = numeric.corr().fillna(0)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Predictive Risk / Driver Correlation")
    plot(fig, height=height)


def lead_to_forecast_funnel(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    marketing = df(data, "marketing")

    if marketing.empty or not all(col in marketing.columns for col in ["impressions", "clicks", "leads", "conversions"]):
        empty_visual("Lead-to-forecast funnel needs marketing funnel data.")
        return

    impressions = safe_sum(marketing, "impressions")
    clicks = safe_sum(marketing, "clicks")
    leads = safe_sum(marketing, "leads")
    conversions = safe_sum(marketing, "conversions")

    close_probability = conversions / leads if leads else 0
    expected_closes = leads * close_probability
    weighted_pipeline = expected_closes * 1.15

    funnel = pd.DataFrame({
        "stage": ["Impressions", "Clicks", "Leads", "Expected closes", "Weighted forecast"],
        "value": [impressions, clicks, leads, expected_closes, weighted_pipeline],
    })

    fig = px.funnel(funnel, x="value", y="stage", title="Lead-to-Forecast Funnel")
    plot(fig, height=height)


def scenario_comparison_chart(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    department = st.session_state.get("forecast_department", "Overview")
    metric = st.session_state.get("forecast_metric", "Revenue")
    periods = int(st.session_state.get("forecast_horizon", 6))
    hr_department = st.session_state.get("forecast_hr_department", "All")

    base = linear_regression_simulation(data, department, metric, periods, 0, 0, 0, 0, hr_department)
    optimistic = linear_regression_simulation(data, department, metric, periods, 12, 6, 8, 5, hr_department)
    conservative = linear_regression_simulation(data, department, metric, periods, -8, -3, -5, -4, hr_department)

    if base.empty or optimistic.empty or conservative.empty:
        empty_visual("Scenario comparison needs Linear Regression baseline data.")
        return

    rows = pd.DataFrame({
        "scenario": ["Baseline", "Optimistic", "Conservative"],
        "forecast_value": [
            float(pd.to_numeric(base["scenario"], errors="coerce").fillna(0).iloc[-1]),
            float(pd.to_numeric(optimistic["scenario"], errors="coerce").fillna(0).iloc[-1]),
            float(pd.to_numeric(conservative["scenario"], errors="coerce").fillna(0).iloc[-1]),
        ],
    })

    fig = px.bar(
        rows,
        x="scenario",
        y="forecast_value",
        text=rows["forecast_value"].apply(lambda x: format_forecast_value(metric, x)),
        title=f"Scenario Comparison • {metric}",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title=metric,
        showlegend=False,
    )

    plot(fig, height=height)

def forecast_confidence_chart(data: Dict[str, pd.DataFrame], height: int = 520, horizon: int = 10) -> None:
    department = st.session_state.get("forecast_department", "Overview")
    metric = st.session_state.get("forecast_metric", "Revenue")
    periods = int(st.session_state.get("forecast_horizon", horizon))

    forecast = linear_regression_forecast(
        data=data,
        department=department,
        metric=metric,
        periods=periods,
        hr_department=st.session_state.get("forecast_hr_department", "All"),
    )

    if forecast.empty or not has_cols(forecast, ["period", "value", "type"]):
        empty_visual("Linear Regression forecast needs historical data from the pipeline.")
        return

    fig = px.line(
        forecast,
        x="period",
        y="value",
        color="type",
        markers=True,
        title=f"{department} {metric} Forecast • Linear Regression",
    )
    fig.update_traces(line=dict(width=2.6), marker=dict(size=6))
    fig.update_layout(legend=dict(orientation="h", y=1.10, x=1, xanchor="right"))
    plot(fig, height=height)


def linear_regression_simulation_chart(data: Dict[str, pd.DataFrame], height: int = 520) -> None:
    department = st.session_state.get("forecast_department", "Overview")
    metric = st.session_state.get("forecast_metric", "Revenue")
    periods = int(st.session_state.get("forecast_horizon", 6))

    scenario = linear_regression_simulation(
        data=data,
        department=department,
        metric=metric,
        periods=periods,
        demand_shift=float(st.session_state.get("sim_demand_shift", 0)),
        price_shift=float(st.session_state.get("sim_price_shift", 0)),
        spend_shift=float(st.session_state.get("sim_spend_shift", 0)),
        capacity_shift=float(st.session_state.get("sim_capacity_shift", 0)),
        hr_department=st.session_state.get("forecast_hr_department", "All"),
    )

    if scenario.empty or not has_cols(scenario, ["period", "baseline", "scenario"]):
        empty_visual("Simulation needs forecast output from the Linear Regression model.")
        return

    scenario = scenario.copy()
    scenario["baseline"] = pd.to_numeric(scenario["baseline"], errors="coerce").fillna(0)
    scenario["scenario"] = pd.to_numeric(scenario["scenario"], errors="coerce").fillna(0)
    scenario["uplift"] = pd.to_numeric(scenario["uplift"], errors="coerce").fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scenario["period"],
        y=scenario["baseline"],
        mode="lines+markers",
        name="Baseline Forecast",
        line=dict(width=2.4, color="#65DCFF"),
    ))

    fig.add_trace(go.Scatter(
        x=scenario["period"],
        y=scenario["scenario"],
        mode="lines+markers",
        name="Scenario Outcome",
        line=dict(width=2.4, dash="dash", color="#37DC8B"),
    ))

    fig.update_layout(
        title=f"{department} {metric} Simulation • Linear Regression Baseline vs Scenario",
        yaxis_title=metric,
        xaxis_title="Forecast Period",
    )

    plot(fig, height=height)

    baseline_final = float(scenario["baseline"].iloc[-1])
    scenario_final = float(scenario["scenario"].iloc[-1])
    uplift_final = float(scenario["uplift"].iloc[-1])
    risk_level = scenario["risk_level"].iloc[-1] if "risk_level" in scenario.columns and not scenario.empty else "Moderate"

    render_html(
        "<div class='forecast-summary-strip'>"
        f"<b>Baseline:</b> {format_forecast_value(metric, baseline_final)} &nbsp;&nbsp; "
        f"<b>Scenario:</b> {format_forecast_value(metric, scenario_final)} &nbsp;&nbsp; "
        f"<b>Uplift:</b> {format_forecast_delta(metric, uplift_final)} &nbsp;&nbsp; "
        f"<b>Risk level:</b> {risk_level} &nbsp;&nbsp; "
        f"<b>Model:</b> Linear Regression baseline with safe scenario adjustment"
        "</div>"
    )

def scenario_comparison_chart(data: Dict[str, pd.DataFrame], height: int = 250) -> None:
    department = st.session_state.get("forecast_department", "Overview")
    metric = st.session_state.get("forecast_metric", "Revenue")
    periods = int(st.session_state.get("forecast_horizon", 6))
    hr_department = st.session_state.get("forecast_hr_department", "All")

    base = linear_regression_simulation(data, department, metric, periods, 0, 0, 0, 0, hr_department)
    optimistic = linear_regression_simulation(data, department, metric, periods, 12, 6, 8, 5, hr_department)
    conservative = linear_regression_simulation(data, department, metric, periods, -8, -3, -5, -4, hr_department)

    if base.empty or optimistic.empty or conservative.empty:
        empty_visual("Scenario comparison needs Linear Regression baseline data.")
        return

    rows = pd.DataFrame({
        "scenario": ["Baseline", "Optimistic", "Conservative"],
        "forecast_value": [
            float(pd.to_numeric(base["scenario"], errors="coerce").fillna(0).iloc[-1]),
            float(pd.to_numeric(optimistic["scenario"], errors="coerce").fillna(0).iloc[-1]),
            float(pd.to_numeric(conservative["scenario"], errors="coerce").fillna(0).iloc[-1]),
        ],
    })

    fig = px.bar(
        rows,
        x="scenario",
        y="forecast_value",
        text=rows["forecast_value"].apply(lambda x: format_forecast_value(metric, x)),
        title=f"Scenario Comparison • {metric}",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title=metric,
        showlegend=False,
    )

    plot(fig, height=height)

def report_source_connectivity_visual(data: Dict[str, pd.DataFrame], height: int = 265) -> None:
    sources = pd.DataFrame({
        "source": ["Sales DB", "CRM", "Marketing Ads", "HR System", "Finance Model", "Report Builder"],
        "x": [0, 0, 0, 0, 0, 1.2],
        "y": [5, 4, 3, 2, 1, 3],
        "records": [len(df(data, "sales")), len(df(data, "customers")), len(df(data, "marketing")), len(df(data, "hr")), 240, 1],
    })
    edges = [(0,5),(1,5),(2,5),(3,5),(4,5)]
    fig = go.Figure()
    for a,b in edges:
        fig.add_trace(go.Scatter(x=[sources.loc[a,"x"],sources.loc[b,"x"]], y=[sources.loc[a,"y"],sources.loc[b,"y"]], mode="lines", line=dict(color="rgba(101,220,255,.35)", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=sources["x"], y=sources["y"], mode="markers+text", text=sources["source"], textposition="middle right",
        marker=dict(size=np.clip(sources["records"].fillna(1)/max(float(sources["records"].max()),1)*34+12, 12, 46), color="#65DCFF", line=dict(color="#F2FBFF", width=1)),
        hovertemplate="%{text}<br>Records: %{marker.size:.0f}<extra></extra>", showlegend=False
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Data Source Connectivity Map")
    plot(fig, height=height)


def report_history_panel() -> None:
    rows = [
        ("Executive Overview", "PDF", "Delivered", "Today 07:12"),
        ("Sales Pipeline", "Excel", "Ready", "Today 06:45"),
        ("Marketing Efficiency", "PDF", "Scheduled", "Yesterday"),
        ("Forecast Pack", "PDF", "Ready", "Yesterday"),
    ]
    html = "<div class='report-card-title'>Completed Report Download History</div>"
    for name, fmt, status, when in rows:
        html += f"<div class='report-history-card'><b>{name}</b><br>{fmt} • {status} • {when}</div>"
    render_html(html)


def distribution_controls_panel() -> None:
    render_html("<div class='report-card-title'>Distribution, Scheduling and Export Controls</div>")
    export_format = st.selectbox("Export format", ["PDF", "Excel", "PowerPoint", "CSV Pack"], key="report_export_format")
    frequency = st.selectbox("Delivery frequency", ["Once", "Daily", "Weekly", "Monthly"], key="report_frequency")
    recipients = st.text_input("Distribution list", "executives@cybernova.com", key="report_recipients")
    if st.button("Generate report package", use_container_width=True):
        st.success(f"Report package queued as {export_format} for {frequency.lower()} delivery.")
    render_html(
        f"<div class='distribution-card'>Format: {export_format}<br>Frequency: {frequency}<br>Recipients: {recipients}</div>"
    )


def report_builder_compact(data: Dict[str, pd.DataFrame]) -> None:
    render_html("<div class='report-card-title'>Live Report Configurator and Preview</div>")

    sales = df(data, "sales")
    marketing = df(data, "marketing")
    advertising = df(data, "advertising")
    hr = df(data, "hr")
    customers = df(data, "customers")

    title = st.text_input(
        "Report title",
        "CYBER NOVA BI Executive Decision Report",
        key="report_builder_title",
    )

    control_a, control_b, control_c = st.columns([1, 1, 1], gap="small")

    with control_a:
        report_department = st.selectbox(
            "Report focus",
            ["Executive Overview", "Sales", "Marketing and Advertising", "HR", "Regional Performance"],
            key="report_focus_area",
        )

    with control_b:
        report_region = st.selectbox(
            "Report region",
            ["All"] + sorted(sales["region"].dropna().unique().tolist()) if "region" in sales.columns else ["All"],
            key="report_region_filter",
        )

    with control_c:
        report_kpi = st.selectbox(
            "Primary KPI",
            ["Revenue", "Profit", "ROI", "Conversions", "ROAS", "Win Rate", "Performance Score"],
            key="report_primary_kpi",
        )

    control_d, control_e = st.columns([1.4, 1], gap="small")

    with control_d:
        selected_sections = st.multiselect(
            "Report sections",
            [
                "Executive Summary",
                "KPI Summary",
                "Department Analysis",
                "Regional Analysis",
                "Forecast Summary",
                "Recommendations",
            ],
            default=["Executive Summary", "KPI Summary", "Recommendations"],
            key="report_builder_sections",
        )

    with control_e:
        chart_type = st.selectbox(
            "Report visual emphasis",
            ["Executive Snapshot", "Trend", "Map", "Forecast", "KPI Pack"],
            key="report_builder_chart_type",
        )

    filtered_data = {
        "sales": sales.copy(),
        "marketing": marketing.copy(),
        "advertising": advertising.copy(),
        "hr": hr.copy(),
        "customers": customers.copy(),
        "web": df(data, "web").copy(),
    }

    if report_region != "All":
        for key, frame in filtered_data.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty and "region" in frame.columns:
                filtered_data[key] = frame[frame["region"] == report_region]

    k = executive_kpis(filtered_data)
    ai = make_ai(filtered_data, "Report")

    sales_metrics_data = sales_metrics(filtered_data)
    marketing_metrics_data = marketing_metrics(filtered_data)
    advertising_metrics_data = advertising_metrics(filtered_data)
    hr_metrics_data = hr_metrics(filtered_data)

    report_lines = [
        title,
        "=" * len(title),
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Report focus: {report_department}",
        f"Region: {report_region}",
        f"Primary KPI: {report_kpi}",
        f"Visual emphasis: {chart_type}",
        "",
    ]

    if "Executive Summary" in selected_sections:
        report_lines.extend([
            "EXECUTIVE SUMMARY",
            str(ai.get("summary", "No executive summary available.")),
            "",
        ])

    if "KPI Summary" in selected_sections:
        report_lines.extend([
            "KPI SUMMARY",
            f"Revenue: {money(k.get('revenue', 0))}",
            f"Profit: {money(k.get('profit', 0))}",
            f"ROI: {pct(k.get('roi', 0))}",
            f"Conversions: {float(k.get('conversions', 0)):,.0f}",
            "",
        ])

    if "Department Analysis" in selected_sections:
        report_lines.append("DEPARTMENT ANALYSIS")

        if report_department in ["Executive Overview", "Sales"]:
            report_lines.extend([
                f"Sales Pipeline Value: {money(sales_metrics_data.get('pipeline_value', 0))}",
                f"Sales Win Rate: {pct(sales_metrics_data.get('win_rate', 0))}",
                f"Average Deal Size: {money(sales_metrics_data.get('avg_deal_size', 0))}",
                "",
            ])

        if report_department in ["Executive Overview", "Marketing and Advertising"]:
            report_lines.extend([
                f"Marketing Leads: {float(marketing_metrics_data.get('leads', 0)):,.0f}",
                f"Marketing Conversion Rate: {pct(marketing_metrics_data.get('conversion_rate', 0))}",
                f"Advertising ROAS: {float(advertising_metrics_data.get('roas', 0)):.2f}x",
                f"Advertising CPA: {money(advertising_metrics_data.get('cpa', 0))}",
                "",
            ])

        if report_department in ["Executive Overview", "HR"]:
            report_lines.extend([
                f"Average Performance: {float(hr_metrics_data.get('avg_performance', 0)):.2f}/5",
                f"Attendance: {pct(hr_metrics_data.get('avg_attendance', 0))}",
                f"Training Hours: {float(hr_metrics_data.get('training_hours', 0)):,.0f}",
                f"High Risk Staff: {int(hr_metrics_data.get('high_risk', 0))}",
                "",
            ])

    if "Regional Analysis" in selected_sections:
        regions = region_summary(filtered_data)
        report_lines.append("REGIONAL ANALYSIS")

        if not regions.empty:
            top_region = regions.iloc[0]
            report_lines.extend([
                f"Strongest Region: {top_region.get('region', 'N/A')}",
                f"Opportunity Score: {float(top_region.get('opportunity_score', 0)):.1f}",
                f"Regional Revenue: {money(top_region.get('revenue', 0))}",
                f"Market Status: {top_region.get('market_status', 'N/A')}",
                "",
            ])
        else:
            report_lines.extend([
                "No regional data available for the selected report filters.",
                "",
            ])

    if "Forecast Summary" in selected_sections:
        forecast = linear_regression_forecast(
            filtered_data,
            department="Overview" if report_department == "Executive Overview" else report_department,
            metric=report_kpi,
            periods=6,
            hr_department="All",
        )

        report_lines.append("FORECAST SUMMARY")

        if not forecast.empty and "type" in forecast.columns:
            forecast_rows = forecast[forecast["type"] == "Forecast"]
            if not forecast_rows.empty:
                final_forecast = float(forecast_rows["value"].iloc[-1])
                accuracy = float(forecast["accuracy"].iloc[-1]) if "accuracy" in forecast.columns else 0
                report_lines.extend([
                    f"Forecast Metric: {report_kpi}",
                    f"Projected Value: {format_forecast_value(report_kpi, final_forecast)}",
                    f"Model Confidence: {accuracy:.1f}%",
                    "",
                ])
            else:
                report_lines.extend([
                    "Forecast output was not available for the selected report configuration.",
                    "",
                ])
        else:
            report_lines.extend([
                "Forecast output was not available for the selected report configuration.",
                "",
            ])

    if "Recommendations" in selected_sections:
        report_lines.append("RECOMMENDED ACTIONS")
        for action in ai.get("actions", []):
            report_lines.append(f"- {action}")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    st.text_area(
        "Report preview",
        report_text,
        height=260,
        key="report_builder_preview",
    )

    st.download_button(
        "Download filtered report",
        report_text,
        file_name=f"cyber_nova_{report_department.lower().replace(' ', '_')}_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="report_builder_download",
    )

    k = executive_kpis(data)
    lines = [
        title,
        "=" * len(title),
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Primary chart: {chart_type}",
        "",
        "SECTIONS",
        ", ".join(selected_sections),
        "",
        "EXECUTIVE SUMMARY",
        str(ai["summary"]),
        "",
        "KPI SUMMARY",
        f"Revenue: {money(k.get('revenue', 0))}",
        f"Profit: {money(k.get('profit', 0))}",
        f"ROI: {pct(k.get('roi', 0))}",
        "",
        "RECOMMENDED ACTIONS",
    ]
    for action in ai["actions"]:
        lines.append(f"- {action}")

    report_text = "\n".join(lines)
    st.text_area("Report preview", report_text, height=260, key="report_builder_preview")
    st.download_button(
        "Download report outline",
        report_text,
        file_name="nexus_bi_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="report_builder_download",
    )


def render_standard_visuals(page: str, data: Dict[str, pd.DataFrame]) -> None:
    with st.container(key="visual_grid"):
        if page == "Sales":
            widths = [1.32, 1.88, 1.32, 1.04]
        elif page == "Marketing and Advertising":
            widths = [1.28, 1.88, 1.32, 1.08]
        elif page == "HR":
            widths = [1.28, 1.88, 1.28, 1.12]
        else:
            widths = [1.16, 1.86, 1.18, 1.20]
        left_col, map_col, right_col, insight_col = st.columns(widths, gap="small")

        with left_col:
            with st.container(key="visual_stack_left"):
                if page == "Overview":
                    with st.container(key="visual_1"):
                        revenue_trend(data, height=250)
                    with st.container(key="visual_2"):
                        regional_score_visual(data, height=250)
                elif page == "Sales":
                    with st.container(key="visual_1"):
                        sales_velocity_treemap(data, height=250)
                    with st.container(key="visual_2"):
                        sales_team_driver_radar(data, height=250)
                elif page == "Marketing and Advertising":
                    with st.container(key="visual_1"):
                        marketing_circular_funnel(data, height=250)
                    with st.container(key="visual_2"):
                        marketing_surface_map(data, height=250)
                elif page == "HR":
                    with st.container(key="visual_1"):
                        hr_parallel_coordinates(data, height=250)
                    with st.container(key="visual_2"):
                        hr_sentiment_gauge(data, height=250)
                else:
                    with st.container(key="visual_1"):
                        revenue_trend(data, height=250)
                    with st.container(key="visual_2"):
                        regional_score_visual(data, height=250)

        with map_col:
            with st.container(key="main_visual"):
                region_map(data, height=520)

        with right_col:
            with st.container(key="visual_stack_right"):
                if page == "Overview":
                    with st.container(key="visual_3"):
                        channel_mix(data, height=250)
                    with st.container(key="visual_4"):
                        performance_opportunity_matrix(data, height=250)
                elif page == "Sales":
                    with st.container(key="visual_3"):
                        sales_deal_stage_funnel(data, height=250)
                    with st.container(key="visual_4"):
                        sales_driver_correlation_matrix(data, height=250)
                elif page == "Marketing and Advertising":
                    with st.container(key="visual_3"):
                        campaign_heatmap(data, height=250)
                    with st.container(key="visual_4"):
                        marketing_spend_waterfall(data, height=250)
                elif page == "HR":
                    with st.container(key="visual_3"):
                        hr_training(data, height=250)
                    with st.container(key="visual_4"):
                        hr_capacity_risk_line(data, height=250)
                else:
                    with st.container(key="visual_3"):
                        channel_mix(data, height=250)
                    with st.container(key="visual_4"):
                        performance_opportunity_matrix(data, height=250)

        with insight_col:
            with st.container(key="insight_panel"):
                assistant_panel(data, page)


def render_forecast_workspace(data: Dict[str, pd.DataFrame]) -> None:
    params = current_simulation_params()
    mode = st.session_state.get("forecast_mode", "Forecast")

    with st.container(key="visual_grid"):
        main_col, side_col, insight_col = st.columns([2.18, 1.22, 1.18], gap="small")

        with main_col:
            with st.container(key="forecast_main"):
                if mode == "Simulation":
                    linear_regression_simulation_chart(data, height=510)
                else:
                    forecast_confidence_chart(data, height=510, horizon=params["horizon"])

        with side_col:
            with st.container(key="forecast_side_stack"):
                with st.container(key="forecast_card_1"):
                    forecast_driver_waterfall(data, height=150)
                with st.container(key="forecast_card_2"):
                    predictive_risk_heatmap(data, height=150)
                with st.container(key="forecast_card_3"):
                    scenario_comparison_chart(data, height=150)

        with insight_col:
            with st.container(key="insight_panel"):
                assistant_panel(data, "Forecast and Simulation Page")


def render_report_workspace(data: Dict[str, pd.DataFrame]) -> None:
    with st.container(key="visual_grid"):
        builder_col, context_col, right_col = st.columns([2.36, 1.10, 1.05], gap="small")
        with builder_col:
            with st.container(key="report_main"):
                report_builder_compact(data)
        with context_col:
            with st.container(key="report_context_stack"):
                with st.container(key="report_source_panel"):
                    report_source_connectivity_visual(data, height=190)
                with st.container(key="report_history_panel"):
                    report_history_panel()
        with right_col:
            with st.container(key="report_right"):
                distribution_controls_panel()


def render_page_visuals(page: str, data: Dict[str, pd.DataFrame]) -> None:
    if page == "Forecast and Simulation Page":
        render_forecast_workspace(data)
    elif page == "Report":
        render_report_workspace(data)
    else:
        render_standard_visuals(page, data)



# DATA & OPTIONS


initial_cleaned, _, _ = run_pipeline()
sales_options = df(initial_cleaned, "sales")
customer_options = df(initial_cleaned, "customers")
hr_options = df(initial_cleaned, "hr")

countries = ["All"] + sorted(sales_options["country"].dropna().unique().tolist()) if "country" in sales_options.columns else ["All"]
regions = ["All"] + sorted(sales_options["region"].dropna().unique().tolist()) if "region" in sales_options.columns else ["All"]
segments = ["All"] + sorted(customer_options["segment"].dropna().unique().tolist()) if "segment" in customer_options.columns else ["All"]
services = ["All"] + sorted(sales_options["service"].dropna().unique().tolist()) if "service" in sales_options.columns else ["All"]
forecast_departments = ["Overview", "Sales", "Marketing and Advertising", "HR"]
hr_forecast_departments = ["All"] + sorted(hr_options["department"].dropna().unique().tolist()) if "department" in hr_options.columns else ["All"]

st.session_state.setdefault("filter_country", countries[0])
st.session_state.setdefault("filter_region", regions[0])
st.session_state.setdefault("filter_period", "Last 30 days")
st.session_state.setdefault("filter_segment", segments[0])
st.session_state.setdefault("filter_service", services[0])



# TOP BAR
page = render_top_bar()

live_mode = st.session_state.live_mode
refresh_speed = st.session_state.refresh_speed
rows_per_tick = st.session_state.rows_per_tick

# LIVE UPDATE
if live_mode:
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_speed * 1000, key="nexus_live_refresh")
    else:
        st.warning("Install streamlit-autorefresh to enable automatic live updates.")

    generate_data.generate_tick(rows_per_tick=rows_per_tick, history=False)

    st.session_state.tick_count += 1
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

    st.session_state.live_log = (
        [f"{st.session_state.last_update} • added {rows_per_tick} rows"]
        + st.session_state.live_log
    )[:8]


# LOAD DATA AND APPLY CURRENT FILTERS

cleaned, validation_report, _ = run_pipeline()

country = st.session_state.get("filter_country", "All")
region = st.session_state.get("filter_region", "All")
period = st.session_state.get("filter_period", "Last 30 days")
segment = st.session_state.get("filter_segment", "All")
service = st.session_state.get("filter_service", "All")

filtered = apply_filters(
    cleaned,
    country=country,
    region=region,
    period=period,
    segment=segment,
    service=service,
)

drilldown_views = build_drilldown_views(filtered)

# KPI ROW
render_kpi_row(filtered, page)

if page == "Forecast and Simulation Page":
    render_forecast_control_bar(forecast_departments, hr_forecast_departments)

# MAIN GRID
with st.container(key="main_workspace"):
    filter_col, visual_area = st.columns([0.98, 5.42], gap="small")

    with filter_col:
        country, region, period, segment, service = render_filter_panel(countries, regions, segments, services, page)

    with visual_area:
        render_page_visuals(page, filtered)

# MOVING INFORMATION BAR
with st.container(key="moving_info_bar"):
    moving_banner(filtered)
