from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Global REIT Dashboard", layout="wide")

st.title("Global REIT Dashboard")
st.caption("Bloomberg-based dashboard for global REIT overview, macro comparison, dividend yield, regional performance, and sector performance")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_first_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def to_numeric_safe(df: pd.DataFrame, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_base_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s

    base = valid.iloc[0]
    if pd.isna(base) or base == 0:
        return s

    return s / base * 100


def format_metric_value(value, unit=None):
    if pd.isna(value) or str(value).strip().lower() in ["none", "nan", ""]:
        return "-"

    if unit and str(unit).strip().lower() not in ["none", "nan", ""]:
        if str(unit).strip() == "%":
            return f"{value}%"
        return f"{value} {unit}"

    return str(value)


def get_overview_metric_map(overview_df: pd.DataFrame):
    metric_map = {}

    if {"metric", "value"}.issubset(set(overview_df.columns)):
        for _, row in overview_df.iterrows():
            metric_name = str(row.get("metric", "")).strip().lower()
            metric_map[metric_name] = {
                "value": row.get("value"),
                "unit": row.get("unit")
            }

    return metric_map


def pick_date_column(df: pd.DataFrame):
    for c in ["date", "datetime", "month"]:
        if c in df.columns:
            return c
    return None


def guess_ust10y_col(columns):
    preferred_keywords = [
        "ust_10y", "ust10y", "us10y", "us_10y", "10y_treasury",
        "10y treasury", "treasury_10y", "treasury10y", "us treasury 10y",
        "u.s. treasury 10y", "ust 10y", "ust 10yr", "10yr treasury",
        "10-year treasury", "10 year treasury", "usgg10yr", "dgs10"
    ]

    cols_lower = {c.lower(): c for c in columns}

    for key in preferred_keywords:
        for c_lower, orig in cols_lower.items():
            if key in c_lower:
                return orig

    for c in columns:
        cl = c.lower()
        if ("treasury" in cl or "ust" in cl or "govt" in cl or "government" in cl) and "10" in cl:
            return c

    return None


def guess_index_columns(columns, exclude=None, max_n=3):
    exclude = set(exclude or [])
    candidates = []

    preferred_keywords = [
        "ftse epra", "nareit", "reit", "reits",
        "s&p 500", "sp500", "s&p",
        "msci world", "msci_world", "msci",
        "global equity", "equity", "stocks",
        "acwi", "ftse", "nikkei", "kospi"
    ]

    for c in columns:
        if c in exclude:
            continue

        cl = c.lower()

        if any(x in cl for x in ["ust", "treasury", "yield", "rate", "cpi", "inflation", "spread", "policy"]):
            continue

        score = 0
        for kw in preferred_keywords:
            if kw in cl:
                score += 1

        if score > 0:
            candidates.append((score, c))

    candidates = [c for _, c in sorted(candidates, key=lambda x: (-x[0], x[1]))]

    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) >= max_n:
        return unique[:max_n]

    for c in columns:
        if c in exclude or c in unique:
            continue

        cl = c.lower()

        if any(x in cl for x in ["ust", "treasury", "yield", "rate", "cpi", "inflation", "spread", "policy"]):
            continue

        unique.append(c)

        if len(unique) >= max_n:
            break

    return unique[:max_n]


def filter_by_period(df: pd.DataFrame, date_col: str, period_label: str):
    if period_label == "All":
        return df.copy()

    latest_date = df[date_col].max()
    if pd.isna(latest_date):
        return df.copy()

    if period_label == "YTD":
        start_date = pd.Timestamp(year=latest_date.year, month=1, day=1)
        return df[df[date_col] >= start_date].copy()

    mapping = {
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
        "10Y": pd.DateOffset(years=10),
    }

    offset = mapping.get(period_label)
    if offset is None:
        return df.copy()

    start_date = latest_date - offset
    return df[df[date_col] >= start_date].copy()


def get_latest_from_timeseries(df: pd.DataFrame, date_col: str, value_col: str):
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty:
        return None, None

    latest_row = temp.iloc[-1]
    return latest_row[date_col], latest_row[value_col]


def calc_ytd_return_from_timeseries(df: pd.DataFrame, date_col: str, value_col: str):
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty:
        return None

    end_date = temp[date_col].max()
    ytd_start = pd.Timestamp(year=end_date.year, month=1, day=1)

    ytd_df = temp[temp[date_col] >= ytd_start]

    if ytd_df.empty:
        return None

    start_value = ytd_df[value_col].iloc[0]
    end_value = temp[value_col].iloc[-1]

    if pd.isna(start_value) or start_value == 0:
        return None

    return (end_value / start_value - 1) * 100


def get_best_item(df: pd.DataFrame, name_col: str, return_col: str):
    temp = df[[name_col, return_col]].copy()
    temp[return_col] = pd.to_numeric(temp[return_col], errors="coerce")
    temp = temp.dropna(subset=[name_col, return_col])

    if temp.empty:
        return None

    return temp.sort_values(return_col, ascending=False).iloc[0]


# --------------------------------------------------
# Loaders
# --------------------------------------------------
@st.cache_data
def load_overview_data():
    candidates = [
        PARSED_DIR / "overview_latest.csv",
        DATA_DIR / "overview_latest.csv",
    ]

    file_path = find_first_existing_file(candidates)

    if file_path is None:
        raise FileNotFoundError(f"overview_latest.csv 파일이 없습니다. 경로 후보: {[str(p) for p in candidates]}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path


@st.cache_data
def load_country_weight_data():
    candidates = [
        PARSED_DIR / "country_weight_latest.csv",
        DATA_DIR / "country_weight_latest.csv",
    ]

    file_path = find_first_existing_file(candidates)

    if file_path is None:
        raise FileNotFoundError(f"country_weight_latest.csv 파일이 없습니다. 경로 후보: {[str(p) for p in candidates]}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path


@st.cache_data
def load_macro_data():
    candidates = [
        DATA_DIR / "timeseries_macro.csv",
        DATA_DIR / "macro_timeseries.csv",
        DATA_DIR / "macro.csv",
    ]

    file_path = find_first_existing_file(candidates)

    if file_path is None:
        raise FileNotFoundError(f"매크로 시계열 파일이 없습니다. 경로 후보: {[str(p) for p in candidates]}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path


@st.cache_data
def load_sector_data():
    file_path = DATA_DIR / "sector_returns.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path


@st.cache_data
def load_region_data():
    file_path = DATA_DIR / "region_returns.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path

@st.cache_data
def load_sector_news_data():
    file_path = DATA_DIR / "sector_news.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")
    df = pd.read_csv(file_path)
    df = normalize_columns(df)
    return df, file_path

# --------------------------------------------------
# Custom style
# --------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        margin-bottom: 12px;
        min-height: 110px;
    }
    .metric-label {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .metric-value {
        color: white;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        word-break: break-word;
    }
    .metric-note {
        color: #6B7280;
        font-size: 0.75rem;
        margin-top: 10px;
        line-height: 1.2;
    }
    .section-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Section: Overview
# --------------------------------------------------
def render_overview():
    st.header("Overview")

    left, right = st.columns([1.1, 1])

    with left:
        try:
            overview_df, _ = load_overview_data()

            if overview_df.empty:
                st.info("overview 데이터가 비어 있습니다.")
            else:
                metric_map = get_overview_metric_map(overview_df)

                # Factsheet-based values
                number_of_constituents = format_metric_value(
                    metric_map.get("number of constituents", {}).get("value"),
                    metric_map.get("number of constituents", {}).get("unit")
                )

                net_market_cap = format_metric_value(
                    metric_map.get("net market cap", {}).get("value"),
                    metric_map.get("net market cap", {}).get("unit")
                )

                # Daily Bloomberg values
                macro_df, _ = load_macro_data()
                date_col = pick_date_column(macro_df)

                daily_as_of = None
                daily_ytd_return = None
                latest_dividend_yield = None

                if date_col is not None:
                    if "ftse epra nareit developed" in macro_df.columns:
                        daily_as_of, _ = get_latest_from_timeseries(
                            macro_df,
                            date_col,
                            "ftse epra nareit developed"
                        )

                        daily_ytd_return = calc_ytd_return_from_timeseries(
                            macro_df,
                            date_col,
                            "ftse epra nareit developed"
                        )

                    if "dividend yield" in macro_df.columns:
                        _, latest_dividend_yield = get_latest_from_timeseries(
                            macro_df,
                            date_col,
                            "dividend yield"
                        )

                daily_as_of_text = daily_as_of.strftime("%Y-%m-%d") if daily_as_of is not None else "-"
                daily_ytd_text = f"{daily_ytd_return:.2f}%" if daily_ytd_return is not None else "-"
                dividend_yield_text = f"{latest_dividend_yield:.2f}%" if latest_dividend_yield is not None else "-"

                # Best region / sector
                best_region_text = "-"
                best_sector_text = "-"

                try:
                    region_df, _ = load_region_data()
                    best_region = get_best_item(region_df, "region", "return_ytd")

                    if best_region is not None:
                        best_region_text = f"{best_region['region']} ({best_region['return_ytd']:.1f}%)"
                except Exception:
                    pass

                try:
                    sector_df, _ = load_sector_data()
                    best_sector = get_best_item(sector_df, "sector", "return_ytd")

                    if best_sector is not None:
                        best_sector_text = f"{best_sector['sector']} ({best_sector['return_ytd']:.1f}%)"
                except Exception:
                    pass

                st.caption(
                    f"Daily Bloomberg data as of {daily_as_of_text}. "
                    "Constituents, net market cap and country weight are factsheet-based as of 30 Apr 2026."
                )

                st.markdown('<div class="section-subtitle">Key Metrics</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c5, c6 = st.columns(2)
                c7, _ = st.columns(2)

                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">As of Date</div>
                            <div class="metric-value">{daily_as_of_text}</div>
                            <div class="metric-note">Daily Bloomberg data</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">YTD Return</div>
                            <div class="metric-value">{daily_ytd_text}</div>
                            <div class="metric-note">FTSE EPRA Nareit Developed</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Dividend Yield</div>
                            <div class="metric-value">{dividend_yield_text}</div>
                            <div class="metric-note">Daily Bloomberg data</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Number of Constituents</div>
                            <div class="metric-value">{number_of_constituents}</div>
                            <div class="metric-note">Factsheet as of 30 Apr 2026</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c5:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Best Region YTD</div>
                            <div class="metric-value">{best_region_text}</div>
                            <div class="metric-note">Regional index performance</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c6:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Best Sector YTD</div>
                            <div class="metric-value">{best_sector_text}</div>
                            <div class="metric-note">Sector index performance</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c7:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Net Market Cap</div>
                            <div class="metric-value">{net_market_cap}</div>
                            <div class="metric-note">Factsheet as of 30 Apr 2026</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.warning("overview 데이터를 불러오지 못했습니다.")
            st.code(str(e))

    with right:
        try:
            country_df, _ = load_country_weight_data()

            if country_df.empty:
                st.info("country weight 데이터가 비어 있습니다.")
            else:
                country_col = next((c for c in country_df.columns if c in ["country", "nation", "region"]), None)
                weight_col = next((c for c in country_df.columns if c in ["weight", "weight_pct", "portfolio_weight", "allocation"]), None)

                if country_col and weight_col:
                    plot_df = country_df.copy()
                    plot_df[weight_col] = pd.to_numeric(plot_df[weight_col], errors="coerce")
                    plot_df = plot_df.dropna(subset=[weight_col]).sort_values(weight_col, ascending=False)

                    st.markdown('<div class="section-subtitle">Country Weight</div>', unsafe_allow_html=True)
                    st.caption("Factsheet as of 30 Apr 2026")

                    fig = px.pie(
                        plot_df,
                        names=country_col,
                        values=weight_col,
                        hole=0.55,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    fig.update_layout(
                        height=500,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("country / weight 컬럼을 찾지 못했습니다.")
        except Exception as e:
            st.warning("country_weight_latest.csv를 불러오지 못했습니다.")
            st.code(str(e))


# --------------------------------------------------
# Section: Macro Comparison
# --------------------------------------------------
def render_macro():
    st.header("Macro Comparison")

    try:
        macro_df, _ = load_macro_data()
    except Exception as e:
        st.warning("매크로 데이터 파일을 불러오지 못했습니다.")
        st.code(str(e))
        return

    if macro_df.empty:
        st.info("매크로 데이터가 비어 있습니다.")
        return

    date_col = pick_date_column(macro_df)

    if date_col is None:
        st.warning("date / datetime / month 컬럼을 찾지 못했습니다.")
        st.dataframe(macro_df, use_container_width=True)
        return

    macro_df = macro_df.copy()
    macro_df[date_col] = pd.to_datetime(macro_df[date_col], errors="coerce")
    macro_df = macro_df.dropna(subset=[date_col]).sort_values(date_col)

    numeric_cols = []

    for c in macro_df.columns:
        if c == date_col:
            continue

        macro_df[c] = pd.to_numeric(macro_df[c], errors="coerce")

        if macro_df[c].notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        st.info("차트로 그릴 숫자형 컬럼이 없습니다.")
        st.dataframe(macro_df, use_container_width=True)
        return

    ust_col = guess_ust10y_col(numeric_cols)

    index_default_cols = guess_index_columns(
        numeric_cols,
        exclude=[ust_col] if ust_col else [],
        max_n=3
    )

    control_left, control_mid = st.columns([3, 1])

    with control_left:
        selected_index_cols = st.multiselect(
            "Select 3 index series",
            options=[c for c in numeric_cols if c != ust_col],
            default=index_default_cols[:3],
            max_selections=3
        )

    with control_mid:
        period = st.selectbox(
            "Period",
            options=["YTD", "1Y", "3Y", "5Y", "10Y", "All"],
            index=0
        )

    if not selected_index_cols:
        st.info("최소 1개 이상의 지수 시리즈를 선택해주세요.")
        return

    plot_cols = [date_col] + selected_index_cols

    use_ust = ust_col is not None

    if use_ust:
        plot_cols.append(ust_col)

    plot_df = macro_df[plot_cols].copy()
    plot_df = filter_by_period(plot_df, date_col, period)

    if plot_df.empty:
        st.info("선택한 기간에 해당하는 데이터가 없습니다.")
        return

    latest_date = plot_df[date_col].max()
    st.caption(f"Daily Bloomberg and FRED data as of {latest_date.strftime('%Y-%m-%d')}")

    line_df = plot_df[[date_col] + selected_index_cols].copy()

    for col in selected_index_cols:
        line_df[col] = to_base_100(line_df[col])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    line_colors = ["#60A5FA", "#34D399", "#F59E0B"]

    for i, col in enumerate(selected_index_cols):
        fig.add_trace(
            go.Scatter(
                x=line_df[date_col],
                y=line_df[col],
                mode="lines",
                name=col,
                line=dict(width=2.7, color=line_colors[i % len(line_colors)])
            ),
            secondary_y=False
        )

    if use_ust:
        ust_plot = plot_df[[date_col, ust_col]].dropna().copy()

        fig.add_trace(
            go.Bar(
                x=ust_plot[date_col],
                y=ust_plot[ust_col],
                name="UST 10Y",
                marker_color="rgba(156,163,175,0.45)",
                opacity=0.55
            ),
            secondary_y=True
        )

    fig.update_layout(
        title="Macro Comparison",
        height=580,
        hovermode="x unified",
        barmode="overlay",
        legend_title="Series",
        margin=dict(t=60, b=20, l=20, r=20)
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Index (Start = 100)", secondary_y=False)

    if use_ust:
        fig.update_yaxes(title_text="UST 10Y Yield (%)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="", secondary_y=True, showgrid=False)

    fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Section: Dividend Yield Trend
# --------------------------------------------------
def render_dividend_yield():
    st.header("Dividend Yield Trend")

    try:
        macro_df, _ = load_macro_data()
    except Exception as e:
        st.warning("배당수익률 데이터를 불러오지 못했습니다.")
        st.code(str(e))
        return

    if macro_df.empty:
        st.info("배당수익률 데이터가 비어 있습니다.")
        return

    date_col = pick_date_column(macro_df)

    if date_col is None:
        st.warning("date / datetime / month 컬럼을 찾지 못했습니다.")
        return

    if "dividend yield" not in macro_df.columns:
        st.warning("Dividend Yield 컬럼을 찾지 못했습니다. data_download.py에서 timeseries_macro.csv에 Dividend Yield를 추가했는지 확인해주세요.")
        st.write("현재 컬럼:", macro_df.columns.tolist())
        return

    plot_df = macro_df[[date_col, "dividend yield"]].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
    plot_df["dividend yield"] = pd.to_numeric(plot_df["dividend yield"], errors="coerce")
    plot_df = plot_df.dropna(subset=[date_col, "dividend yield"]).sort_values(date_col)

    if plot_df.empty:
        st.info("차트로 그릴 Dividend Yield 데이터가 없습니다.")
        return

    latest_date = plot_df[date_col].max()
    latest_value = plot_df["dividend yield"].dropna().iloc[-1]

    ten_year_start = latest_date - pd.DateOffset(years=10)
    avg_10y_df = plot_df[plot_df[date_col] >= ten_year_start].copy()
    avg_10y = avg_10y_df["dividend yield"].mean() if not avg_10y_df.empty else None

    period = st.selectbox(
        "Dividend Yield Period",
        options=["YTD", "1Y", "3Y", "5Y", "10Y", "All"],
        index=0
    )

    chart_df = filter_by_period(plot_df, date_col, period)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Latest Dividend Yield",
            value=f"{latest_value:.2f}%"
        )

    with col2:
        if avg_10y is not None:
            st.metric(
                label="10Y Average Dividend Yield",
                value=f"{avg_10y:.2f}%"
            )

    st.caption(f"Daily Bloomberg data as of {latest_date.strftime('%Y-%m-%d')}")

    fig = px.line(
        chart_df,
        x=date_col,
        y="dividend yield",
        title="FTSE EPRA Nareit Developed Dividend Yield",
        markers=False
    )

    if avg_10y is not None:
        fig.add_hline(
            y=avg_10y,
            line_dash="dash",
            annotation_text=f"10Y Avg: {avg_10y:.2f}%",
            annotation_position="top left"
        )

    fig.update_layout(
        height=480,
        xaxis_title="Date",
        yaxis_title="Dividend Yield (%)",
        hovermode="x unified",
        margin=dict(t=60, b=20, l=20, r=20)
    )

    fig.update_traces(line=dict(width=2.7))

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Section: Regional Performance
# --------------------------------------------------
def render_region_performance():
    st.header("Regional Performance")

    try:
        region_df, _ = load_region_data()
    except Exception as e:
        st.warning("region_returns.csv를 불러오지 못했습니다.")
        st.code(str(e))
        return

    if region_df.empty:
        st.info("지역 데이터가 비어 있습니다.")
        return

    required_cols = ["region", "return_1m", "return_ytd", "return_12m"]
    missing_cols = [c for c in required_cols if c not in region_df.columns]

    if missing_cols:
        st.error(f"region_returns.csv에 필요한 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", region_df.columns.tolist())
        return

    region_df = to_numeric_safe(region_df, ["return_1m", "return_ytd", "return_12m"])

    if "as_of_date" in region_df.columns and region_df["as_of_date"].notna().any():
        as_of_date = region_df["as_of_date"].dropna().iloc[0]
        st.caption(f"Daily Bloomberg data as of {as_of_date}")

    tab1, tab2, tab3 = st.tabs(["1 Month", "YTD", "Trailing 12 Months"])

    with tab1:
        df_1m = region_df.dropna(subset=["return_1m"]).sort_values("return_1m", ascending=False)

        if df_1m.empty:
            st.info("1개월 수익률 데이터가 없습니다.")
        else:
            fig_1m = px.bar(
                df_1m,
                x="region",
                y="return_1m",
                color="return_1m",
                color_continuous_scale="Blues",
                text="return_1m"
            )
            fig_1m.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_1m.update_layout(
                height=500,
                xaxis_title="Region",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_1m, use_container_width=True)

    with tab2:
        df_ytd = region_df.dropna(subset=["return_ytd"]).sort_values("return_ytd", ascending=False)

        if df_ytd.empty:
            st.info("YTD 수익률 데이터가 없습니다.")
        else:
            fig_ytd = px.bar(
                df_ytd,
                x="region",
                y="return_ytd",
                color="return_ytd",
                color_continuous_scale="Blues",
                text="return_ytd"
            )
            fig_ytd.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_ytd.update_layout(
                height=500,
                xaxis_title="Region",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_ytd, use_container_width=True)

    with tab3:
        df_12m = region_df.dropna(subset=["return_12m"]).sort_values("return_12m", ascending=False)

        if df_12m.empty:
            st.info("12개월 수익률 데이터가 없습니다.")
        else:
            fig_12m = px.bar(
                df_12m,
                x="region",
                y="return_12m",
                color="return_12m",
                color_continuous_scale="Blues",
                text="return_12m"
            )
            fig_12m.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_12m.update_layout(
                height=500,
                xaxis_title="Region",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_12m, use_container_width=True)


# --------------------------------------------------
# Section: Sector Performance
# --------------------------------------------------
def render_sector_performance():
    st.header("Sector Performance")

    try:
        sector_df, _ = load_sector_data()
    except Exception as e:
        st.warning("sector_returns.csv를 불러오지 못했습니다.")
        st.code(str(e))
        return

    if sector_df.empty:
        st.info("섹터 데이터가 비어 있습니다.")
        return

    required_cols = ["sector", "return_1m", "return_ytd", "return_12m"]
    missing_cols = [c for c in required_cols if c not in sector_df.columns]

    if missing_cols:
        st.error(f"sector_returns.csv에 필요한 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", sector_df.columns.tolist())
        return

    sector_df = to_numeric_safe(sector_df, ["return_1m", "return_ytd", "return_12m", "n_tickers"])

    if "as_of_date" in sector_df.columns and sector_df["as_of_date"].notna().any():
        as_of_date = sector_df["as_of_date"].dropna().iloc[0]
        st.caption(f"Daily Bloomberg data as of {as_of_date}")

    tab1, tab2, tab3 = st.tabs(["1 Month", "YTD", "Trailing 12 Months"])

    with tab1:
        df_1m = sector_df.dropna(subset=["return_1m"]).sort_values("return_1m", ascending=False)

        if df_1m.empty:
            st.info("1개월 수익률 데이터가 없습니다.")
        else:
            fig_1m = px.bar(
                df_1m,
                x="sector",
                y="return_1m",
                color="return_1m",
                color_continuous_scale="Blues",
                text="return_1m"
            )
            fig_1m.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_1m.update_layout(
                height=500,
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_1m, use_container_width=True)

    with tab2:
        df_ytd = sector_df.dropna(subset=["return_ytd"]).sort_values("return_ytd", ascending=False)

        if df_ytd.empty:
            st.info("YTD 수익률 데이터가 없습니다.")
        else:
            fig_ytd = px.bar(
                df_ytd,
                x="sector",
                y="return_ytd",
                color="return_ytd",
                color_continuous_scale="Blues",
                text="return_ytd"
            )
            fig_ytd.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_ytd.update_layout(
                height=500,
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_ytd, use_container_width=True)

    with tab3:
        df_12m = sector_df.dropna(subset=["return_12m"]).sort_values("return_12m", ascending=False)

        if df_12m.empty:
            st.info("12개월 수익률 데이터가 없습니다.")
        else:
            fig_12m = px.bar(
                df_12m,
                x="sector",
                y="return_12m",
                color="return_12m",
                color_continuous_scale="Blues",
                text="return_12m"
            )
            fig_12m.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_12m.update_layout(
                height=500,
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_12m, use_container_width=True)

# --------------------------------------------------
# Section: Sector News & Key Issues
# --------------------------------------------------
def render_sector_news():
    st.header("Sector News & Key Issues")
    st.caption("News collected from Google News RSS using sector keywords and representative REIT tickers.")

    try:
        news_df, _ = load_sector_news_data()
    except Exception as e:
        st.warning("sector_news.csv를 불러오지 못했습니다. 먼저 news_download.py를 실행해주세요.")
        st.code(str(e))
        return

    if news_df.empty:
        st.info("뉴스 데이터가 비어 있습니다.")
        return

    required_cols = ["sector", "title", "source", "published", "summary", "link"]
    missing_cols = [c for c in required_cols if c not in news_df.columns]

    if missing_cols:
        st.error(f"sector_news.csv에 필요한 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", news_df.columns.tolist())
        return

    sectors = news_df["sector"].dropna().unique().tolist()

    selected_sector = st.selectbox(
        "Select Sector",
        options=["All"] + sectors,
        index=0
    )

    if selected_sector != "All":
        display_df = news_df[news_df["sector"] == selected_sector].copy()
    else:
        display_df = news_df.copy()

    max_articles = st.slider(
        "Number of articles to show",
        min_value=3,
        max_value=20,
        value=9,
        step=1
    )

    display_df = display_df.head(max_articles)

    for _, row in display_df.iterrows():
        sector = row.get("sector", "")
        title = row.get("title", "")
        source = row.get("source", "")
        published = row.get("published", "")
        summary = row.get("summary", "")
        link = row.get("link", "")

        if pd.isna(title) or str(title).strip() == "":
            continue

        st.markdown(
            f"""
            <div style="
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 12px;
                padding: 14px 16px;
                margin-bottom: 12px;
                background-color: rgba(17,24,39,0.65);
            ">
                <div style="font-size:0.80rem; color:#9CA3AF; margin-bottom:6px;">
                    {sector} | {source} | {published}
                </div>
                <div style="font-size:1.05rem; font-weight:700; margin-bottom:8px;">
                    <a href="{link}" target="_blank" style="text-decoration:none;">
                        {title}
                    </a>
                </div>
                <div style="font-size:0.88rem; color:#D1D5DB;">
                    {summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Render
# --------------------------------------------------
render_overview()
st.divider()

render_macro()
st.divider()

render_dividend_yield()
st.divider()

render_region_performance()
st.divider()

render_sector_performance()
st.divider()

render_sector_news()