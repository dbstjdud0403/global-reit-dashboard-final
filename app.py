
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# yfinance is used only for the supplementary Key Constituents Price Performance section.
try:
    import yfinance as yf
except Exception:
    yf = None


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Global REIT Dashboard", layout="wide")

st.title("Global REIT Dashboard")
st.caption(
    "Global listed REIT dashboard combining Bloomberg-based index data, "
    "FRED API Treasury yield data, yfinance constituent prices, and Google News RSS scraping."
)


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
                "unit": row.get("unit"),
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

    cols_lower = {str(c).lower(): c for c in columns}

    for key in preferred_keywords:
        for c_lower, orig in cols_lower.items():
            if key in c_lower:
                return orig

    for c in columns:
        cl = str(c).lower()
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

        cl = str(c).lower()

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

        cl = str(c).lower()

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
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
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




def get_latest_market_yield_metrics(df: pd.DataFrame, date_col: str):
    """
    Return the latest common-date dividend yield / UST10Y / spread used in
    the Market Yield Spread section. This keeps Overview and Valuation
    Dividend Yield values consistent.
    """
    if date_col is None or "dividend yield" not in df.columns:
        return None

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp["dividend yield"] = pd.to_numeric(temp["dividend yield"], errors="coerce")

    numeric_cols = []
    for c in temp.columns:
        if c == date_col:
            continue
        temp[c] = pd.to_numeric(temp[c], errors="coerce")
        if temp[c].notna().sum() > 0:
            numeric_cols.append(c)

    ust_col = guess_ust10y_col(numeric_cols)
    if ust_col is None:
        div_temp = temp[[date_col, "dividend yield"]].dropna().sort_values(date_col)
        if div_temp.empty:
            return None
        row = div_temp.iloc[-1]
        return {
            "date": row[date_col],
            "dividend_yield": row["dividend yield"],
            "ust10y": None,
            "spread": None,
            "ust_col": None,
        }

    common = temp[[date_col, "dividend yield", ust_col]].dropna().sort_values(date_col)
    if common.empty:
        return None

    row = common.iloc[-1]
    return {
        "date": row[date_col],
        "dividend_yield": row["dividend yield"],
        "ust10y": row[ust_col],
        "spread": row["dividend yield"] - row[ust_col],
        "ust_col": ust_col,
    }


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


def calc_period_return_from_timeseries(df: pd.DataFrame, date_col: str, value_col: str, period_label: str):
    """
    Calculate total return over the selected period using index levels.
    YTD is the default dashboard period.
    """
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty:
        return None

    end_date = temp[date_col].max()

    if period_label == "YTD":
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif period_label == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif period_label == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif period_label == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    elif period_label == "10Y":
        start_date = end_date - pd.DateOffset(years=10)
    else:
        start_date = temp[date_col].min()

    start_candidates = temp[temp[date_col] >= start_date]

    if start_candidates.empty:
        return None

    start_value = start_candidates[value_col].iloc[0]
    end_value = temp[value_col].iloc[-1]

    if pd.isna(start_value) or start_value == 0:
        return None

    return (end_value / start_value - 1) * 100


def get_latest_numeric_value(df: pd.DataFrame, date_col: str, value_col: str):
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty:
        return None

    return temp[value_col].iloc[-1]


@st.cache_data(ttl=60 * 60 * 24)
def get_yfinance_company_name(ticker):
    """Return company name from yfinance. If unavailable, return the ticker."""
    if yf is None or ticker is None or str(ticker).strip() == "":
        return ticker

    ticker = str(ticker).strip()

    try:
        info = yf.Ticker(ticker).get_info()
        name = info.get("shortName") or info.get("longName") or info.get("displayName") or ticker
        return name
    except Exception:
        return ticker


@st.cache_data(ttl=60 * 60 * 24)
def get_yfinance_company_names(tickers):
    name_map = {}
    for ticker in tickers:
        if ticker is None or str(ticker).strip() == "":
            continue
        ticker = str(ticker).strip()
        name_map[ticker] = get_yfinance_company_name(ticker)
    return name_map


def make_ticker_label(ticker, name_map):
    name = name_map.get(ticker, ticker)
    if name == ticker:
        return ticker
    return f"{name} ({ticker})"


def build_basic_pdf_bytes(title, lines):
    """
    Minimal text-only PDF fallback used when reportlab is not installed.
    This keeps the PDF download button functional even before optional reportlab/kaleido setup.
    """
    def esc(text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    page_width, page_height = 612, 792
    x, y = 50, 742
    content = ["BT", "/F1 16 Tf", f"{x} {y} Td", f"({esc(title)}) Tj"]
    y_step = 15
    content.append("/F1 9 Tf")
    content.append(f"0 -{y_step * 2} Td")

    safe_lines = []
    for line in lines:
        line = str(line)
        if len(line) > 105:
            chunks = [line[i:i + 105] for i in range(0, len(line), 105)]
            safe_lines.extend(chunks)
        else:
            safe_lines.append(line)

    current_y = y - y_step * 2
    for line in safe_lines[:42]:
        if current_y < 60:
            break
        content.append(f"({esc(line)}) Tj")
        content.append(f"0 -{y_step} Td")
        current_y -= y_step

    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return bytes(pdf)


def get_best_item(df: pd.DataFrame, name_col: str, return_col: str):
    temp = df[[name_col, return_col]].copy()
    temp[return_col] = pd.to_numeric(temp[return_col], errors="coerce")
    temp = temp.dropna(subset=[name_col, return_col])

    if temp.empty:
        return None

    return temp.sort_values(return_col, ascending=False).iloc[0]


def clean_sector_yield_name(col_name):
    name = str(col_name).strip()
    name = name.replace("_yield", "")
    name = name.replace(" yield", "")
    name = name.replace("_", " ")
    return name.title()


def classify_sector_valuation_view(row):
    gap = row.get("yield_gap", None)
    spread = row.get("spread_vs_us10y", None)

    if pd.isna(gap) or pd.isna(spread):
        return "Insufficient data"

    if gap > 1.0 and spread > 0:
        return "Relatively cheap"
    if gap > 1.0 and spread <= 0:
        return "Cheap vs history, less attractive vs bonds"
    if gap < 0 and spread < 0:
        return "Premium"
    return "Neutral"


def make_short_comment(sector, issue_tag):
    issue = str(issue_tag)

    if "AI / Data Center Demand" in issue:
        return "AI, cloud expansion and power capacity are the key news themes."
    if "Office Vacancy Risk" in issue:
        return "Vacancy pressure, weak leasing demand and refinancing risks are the key themes."
    if "Healthcare / Senior Housing" in issue:
        return "Senior housing demand, operating recovery and earnings momentum are the key themes."
    if "Industrial / Logistics Demand" in issue:
        return "Warehouse leasing, logistics demand and supply chain activity are the key themes."
    if "Retail / Consumer Demand" in issue:
        return "Tenant sales, consumer spending and shopping center traffic are the key themes."
    if "Housing / Rent Growth" in issue:
        return "Rental demand, apartment fundamentals and housing market trends are the key themes."
    if "Rates / Treasury Yield" in issue:
        return "Interest rates, Treasury yields and monetary policy expectations are the key themes."
    if "Earnings / Guidance" in issue:
        return "Earnings results, guidance and operating performance are the key themes."
    if "M&A / Transaction" in issue:
        return "Acquisitions, transactions, IPOs or portfolio activity are the key themes."
    if "Hotels / Travel Demand" in issue:
        return "Travel demand, occupancy and lodging fundamentals are the key themes."
    if "Self Storage Demand" in issue:
        return "Self-storage demand and operating trends are the key themes."
    if "Net Lease / Tenant Credit" in issue:
        return "Net lease fundamentals, tenant credit and lease stability are the key themes."

    return f"Recent news for {sector} is broadly related to general REIT market developments."


def bloomberg_to_yfinance_ticker(bbg_ticker):
    """
    Convert Bloomberg-style tickers into yfinance-compatible tickers.

    Examples:
    DLR UN   -> DLR
    EQIX UW  -> EQIX
    778 HK   -> 0778.HK
    1686 HK  -> 1686.HK
    3287 JT  -> 3287.T
    330590 KP -> 330590.KS
    GMG AT   -> GMG.AX
    LAND LN  -> LAND.L
    """
    if pd.isna(bbg_ticker):
        return None

    ticker = str(bbg_ticker).strip()

    if ticker == "" or ticker.lower() in ["nan", "none", "#n/a"]:
        return None

    parts = ticker.split()

    if len(parts) == 1:
        return ticker

    code = parts[0]
    exch = parts[1].upper()

    us_suffixes = ["UN", "UW", "UQ", "UP", "UA"]
    if exch in us_suffixes:
        return code

    # yfinance Hong Kong tickers generally require 4 digits.
    if exch == "HK" and code.isdigit():
        return f"{code.zfill(4)}.HK"

    suffix_map = {
        "HK": ".HK",
        "JT": ".T",
        "JP": ".T",
        "KP": ".KS",
        "KQ": ".KQ",
        "LN": ".L",
        "AT": ".AX",
        "AU": ".AX",
        "SP": ".SI",
        "SM": ".MC",
        "FP": ".PA",
        "GR": ".DE",
        "SW": ".SW",
        "SS": ".ST",
        "CN": ".TO",
        "BB": ".BR",
        "NA": ".AS",
        "AV": ".VI",
    }

    if exch in suffix_map:
        return f"{code}{suffix_map[exch]}"

    return code

def get_bloomberg_listing_metadata(bbg_ticker):
    """
    Return broad region/country/currency metadata from Bloomberg ticker suffix.
    Used to clarify that constituent prices are normalized and converted to USD.
    """
    if pd.isna(bbg_ticker):
        return {"region": "Unknown", "country": "Unknown", "currency": "USD", "exchange_suffix": ""}

    parts = str(bbg_ticker).strip().split()
    exch = parts[1].upper() if len(parts) > 1 else ""

    meta_map = {
        "UN": ("North America", "United States", "USD"),
        "UW": ("North America", "United States", "USD"),
        "UQ": ("North America", "United States", "USD"),
        "UP": ("North America", "United States", "USD"),
        "UA": ("North America", "United States", "USD"),
        "HK": ("Asia Pacific", "Hong Kong", "HKD"),
        "JT": ("Asia Pacific", "Japan", "JPY"),
        "JP": ("Asia Pacific", "Japan", "JPY"),
        "KP": ("Asia Pacific", "South Korea", "KRW"),
        "KQ": ("Asia Pacific", "South Korea", "KRW"),
        "AT": ("Asia Pacific", "Australia", "AUD"),
        "AU": ("Asia Pacific", "Australia", "AUD"),
        "SP": ("Asia Pacific", "Singapore", "SGD"),
        "LN": ("Europe", "United Kingdom", "GBP"),
        "SM": ("Europe", "Spain", "EUR"),
        "FP": ("Europe", "France", "EUR"),
        "GR": ("Europe", "Germany", "EUR"),
        "SW": ("Europe", "Switzerland", "CHF"),
        "SS": ("Europe", "Sweden", "SEK"),
        "CN": ("North America", "Canada", "CAD"),
    }

    region, country, currency = meta_map.get(exch, ("Unknown", "Unknown", "USD"))
    return {"region": region, "country": country, "currency": currency, "exchange_suffix": exch}


FX_TO_USD_TICKERS = {
    "HKD": "HKDUSD=X",
    "JPY": "JPYUSD=X",
    "KRW": "KRWUSD=X",
    "AUD": "AUDUSD=X",
    "SGD": "SGDUSD=X",
    "GBP": "GBPUSD=X",
    "EUR": "EURUSD=X",
    "CHF": "CHFUSD=X",
    "SEK": "SEKUSD=X",
    "CAD": "CADUSD=X",
}

# Fallback FX rates used only if yfinance FX data is unavailable.
# This prevents local-currency prices from being mislabeled as USD in the constituent table.
STATIC_FX_TO_USD = {
    "USD": 1.0000,
    "HKD": 0.1280,
    "JPY": 0.0064,
    "KRW": 0.00073,
    "AUD": 0.6650,
    "SGD": 0.7400,
    "GBP": 1.2700,
    "EUR": 1.0850,
    "CHF": 1.1100,
    "SEK": 0.0950,
    "CAD": 0.7300,
}


@st.cache_data(ttl=60 * 60 * 6)
def load_fx_to_usd_series(currency, start_date, end_date):
    """Load currency-to-USD FX series from yfinance. USD returns a 1.0 series."""
    if currency == "USD":
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.Series(1.0, index=dates, name="USD")

    if yf is None or currency not in FX_TO_USD_TICKERS:
        return pd.Series(dtype="float64")

    try:
        fx_symbol = FX_TO_USD_TICKERS[currency]
        fx = yf.download(
            fx_symbol,
            start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if fx.empty or "Close" not in fx.columns:
            return pd.Series(dtype="float64")

        s = fx["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.rename(currency)
    except Exception:
        return pd.Series(dtype="float64")


def convert_prices_to_usd(price_df, ticker_currency_map):
    """Convert local-currency price series to USD using yfinance FX rates."""
    if price_df.empty:
        return price_df.copy()

    usd_df = price_df.copy()
    usd_df.index = pd.to_datetime(usd_df.index).tz_localize(None)

    start_date = usd_df.index.min()
    end_date = usd_df.index.max()

    for ticker in usd_df.columns:
        currency = ticker_currency_map.get(ticker, "USD")

        if currency == "USD":
            continue

        fx_series = load_fx_to_usd_series(currency, start_date, end_date)

        if fx_series.empty:
            fallback_rate = STATIC_FX_TO_USD.get(currency)
            if fallback_rate is None:
                # If no FX source is available, leave the series unchanged rather than dropping it.
                continue
            usd_df[ticker] = usd_df[ticker] * fallback_rate
            continue

        fx_aligned = fx_series.reindex(usd_df.index).ffill().bfill()
        usd_df[ticker] = usd_df[ticker] * fx_aligned

    return usd_df


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

    # Normalize common country aliases from FTSE factsheet parsing.
    # FTSE often uses "USA" while the dashboard labels it as "United States".
    country_col = next((c for c in df.columns if c in ["country", "nation", "region"]), None)
    if country_col is not None:
        alias_map = {
            "usa": "United States",
            "u.s.a.": "United States",
            "u.s.": "United States",
            "united states": "United States",
            "uk": "United Kingdom",
            "u.k.": "United Kingdom",
        }
        df[country_col] = df[country_col].astype(str).str.strip().apply(
            lambda x: alias_map.get(x.lower(), x)
        )
        weight_col = next((c for c in df.columns if c in ["weight", "weight_pct", "portfolio_weight", "allocation"]), None)
        if weight_col is not None:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
            df = df.groupby(country_col, as_index=False)[weight_col].sum()

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
def load_sector_yield_data():
    candidates = [
        DATA_DIR / "sector_yields.csv",
        DATA_DIR / "sector_dividend_yields.csv",
    ]

    file_path = find_first_existing_file(candidates)

    if file_path is None:
        raise FileNotFoundError(f"sector_yields.csv 파일이 없습니다. 경로 후보: {[str(p) for p in candidates]}")

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    return df, file_path


@st.cache_data
def load_sector_ticker_universe():
    file_path = DATA_DIR / "sector_ticker_universe.csv"

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


@st.cache_data(ttl=60 * 60 * 6)
def load_yfinance_prices(tickers, period="YTD"):
    if yf is None:
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

    tickers = list(dict.fromkeys([t for t in tickers if t]))

    if not tickers:
        return pd.DataFrame()

    yf_period = "1y" if period == "YTD" else period

    try:
        data = yf.download(
            tickers=tickers,
            period=yf_period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        if "Close" in data.columns:
            close_df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            return pd.DataFrame()
    else:
        close_dict = {}
        for t in tickers:
            try:
                close_dict[t] = data[t]["Close"]
            except Exception:
                continue
        close_df = pd.DataFrame(close_dict)

    close_df = close_df.dropna(how="all")

    if period == "YTD" and not close_df.empty:
        idx = pd.to_datetime(close_df.index, errors="coerce")
        close_df.index = idx
        latest_date = close_df.index.max()
        if pd.notna(latest_date):
            ytd_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
            close_df = close_df[close_df.index >= ytd_start]

    return close_df


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
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Download report helpers
# --------------------------------------------------
def build_market_summary_text():
    lines = []
    lines.append("Global REIT Dashboard Summary")
    lines.append("=" * 40)

    try:
        macro_df, _ = load_macro_data()
        date_col = pick_date_column(macro_df)

        if date_col is not None:
            macro_df = macro_df.copy()
            macro_df[date_col] = pd.to_datetime(macro_df[date_col], errors="coerce")
            macro_df = macro_df.dropna(subset=[date_col]).sort_values(date_col)

            latest_date = macro_df[date_col].max()
            lines.append(f"As of Date: {latest_date.strftime('%Y-%m-%d')}")

            if "ftse epra nareit developed" in macro_df.columns:
                ytd = calc_ytd_return_from_timeseries(macro_df, date_col, "ftse epra nareit developed")
                if ytd is not None:
                    lines.append(f"Global REIT YTD Return: {ytd:.2f}%")

            div_yield = None
            if "dividend yield" in macro_df.columns:
                _, div_yield = get_latest_from_timeseries(macro_df, date_col, "dividend yield")
                if div_yield is not None:
                    lines.append(f"Global REIT Dividend Yield: {div_yield:.2f}%")

            numeric_cols = []
            for c in macro_df.columns:
                if c == date_col:
                    continue
                macro_df[c] = pd.to_numeric(macro_df[c], errors="coerce")
                if macro_df[c].notna().sum() > 0:
                    numeric_cols.append(c)

            ust_col = guess_ust10y_col(numeric_cols)
            if ust_col is not None:
                _, ust = get_latest_from_timeseries(macro_df, date_col, ust_col)
                if ust is not None:
                    lines.append(f"UST 10Y Yield (FRED DGS10 API): {ust:.2f}%")
                    if div_yield is not None:
                        lines.append(f"REIT Yield Spread: {div_yield - ust:.2f}%p")

    except Exception:
        lines.append("Market summary: unavailable")

    try:
        region_df, _ = load_region_data()
        best_region = get_best_item(region_df, "region", "return_ytd")
        if best_region is not None:
            lines.append(f"Best Region YTD: {best_region['region']} ({best_region['return_ytd']:.2f}%)")
    except Exception:
        pass

    try:
        sector_df, _ = load_sector_data()
        best_sector = get_best_item(sector_df, "sector", "return_ytd")
        if best_sector is not None:
            lines.append(f"Best Sector YTD: {best_sector['sector']} ({best_sector['return_ytd']:.2f}%)")
    except Exception:
        pass

    try:
        news_df, _ = load_sector_news_data()
        if not news_df.empty and "issue_tag" in news_df.columns:
            tag_series = news_df["issue_tag"].fillna("General").astype(str).str.split(", ").explode()
            if not tag_series.empty:
                lines.append(f"Most Mentioned News Issue: {tag_series.value_counts().index[0]}")
    except Exception:
        pass

    lines.append("")
    lines.append("Generated from the Streamlit Global REIT Dashboard.")
    return "\n".join(lines)


def make_ytd_return_bar(df, name_col, return_col, title):
    plot_df = df.copy()
    plot_df[return_col] = pd.to_numeric(plot_df[return_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[name_col, return_col]).sort_values(return_col, ascending=False)
    if plot_df.empty:
        return None
    fig = px.bar(plot_df, x=name_col, y=return_col, text=return_col, title=title)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=460, xaxis_title=name_col.title(), yaxis_title="Return (%)", margin=dict(t=60, b=50, l=40, r=20))
    return fig


def make_market_yield_spread_fig():
    macro_df, _ = load_macro_data()
    date_col = pick_date_column(macro_df)
    if date_col is None or "dividend yield" not in macro_df.columns:
        return None
    macro_df = macro_df.copy()
    macro_df[date_col] = pd.to_datetime(macro_df[date_col], errors="coerce")
    macro_df = macro_df.dropna(subset=[date_col]).sort_values(date_col)
    for c in macro_df.columns:
        if c != date_col:
            macro_df[c] = pd.to_numeric(macro_df[c], errors="coerce")
    ust_col = guess_ust10y_col([c for c in macro_df.columns if c != date_col])
    if ust_col is None:
        return None
    plot_df = macro_df[[date_col, "dividend yield", ust_col]].dropna().copy()
    if plot_df.empty:
        return None
    plot_df["yield spread"] = plot_df["dividend yield"] - plot_df[ust_col]
    plot_df = filter_by_period(plot_df, date_col, "YTD")
    fig = px.line(plot_df, x=date_col, y="yield spread", title="REIT Yield Spread Trend (YTD)")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Spread (%p)", margin=dict(t=60, b=40, l=40, r=20))
    return fig


def make_sector_current_yield_fig():
    yield_df, _ = load_sector_yield_data()
    date_col = pick_date_column(yield_df)
    if date_col is None:
        return None
    yield_df = yield_df.copy()
    yield_df[date_col] = pd.to_datetime(yield_df[date_col], errors="coerce")
    yield_df = yield_df.dropna(subset=[date_col]).sort_values(date_col)
    sector_cols = [c for c in yield_df.columns if c != date_col]
    for c in sector_cols:
        yield_df[c] = pd.to_numeric(yield_df[c], errors="coerce")
    sector_cols = [c for c in sector_cols if yield_df[c].notna().sum() > 0]
    if not sector_cols:
        return None
    latest = yield_df.dropna(subset=sector_cols, how="all").iloc[-1]
    summary = pd.DataFrame({"sector": [clean_sector_yield_name(c) for c in sector_cols], "current_yield": latest[sector_cols].values})
    summary = summary.sort_values("current_yield", ascending=False)
    fig = px.bar(summary, x="sector", y="current_yield", text="current_yield", title="Current Dividend Yield by Sector")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=460, xaxis_title="Sector", yaxis_title="Dividend Yield (%)", margin=dict(t=60, b=50, l=40, r=20))
    return fig


def add_plotly_figure_to_pdf_story(story, fig, title, width_in=7.0, height_in=3.7):
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Image, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()
    story.append(Paragraph(title, styles["Heading2"]))
    if fig is None:
        story.append(Paragraph("Chart unavailable.", styles["BodyText"]))
        story.append(Spacer(1, 8))
        return
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=1100, height=620, scale=2)
        story.append(Image(BytesIO(img_bytes), width=width_in * inch, height=height_in * inch))
        story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"Chart image unavailable: {e}", styles["BodyText"]))
        story.append(Spacer(1, 8))


def build_dashboard_pdf_report():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    except Exception:
        fallback_lines = build_market_summary_text().split("\n")
        fallback_lines.insert(0, "ReportLab is not installed, so this is a text-only PDF fallback.")
        fallback_lines.insert(1, "For chart images in the PDF, add reportlab and kaleido to requirements.txt.")
        return build_basic_pdf_bytes("Global REIT Dashboard Report", fallback_lines)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Global REIT Dashboard Report", styles["Title"]))
    story.append(Paragraph("YTD default view. Data combines Bloomberg index data, FRED DGS10 API, yfinance, and Google News RSS scraping.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    summary_lines = build_market_summary_text().split("\n")
    for line in summary_lines:
        if line.strip() and not set(line.strip()) == {"="}:
            story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 12))

    try:
        region_df, _ = load_region_data()
        fig_region = make_ytd_return_bar(region_df, "region", "return_ytd", "Regional Performance - YTD")
    except Exception:
        fig_region = None
    add_plotly_figure_to_pdf_story(story, fig_region, "Regional Performance - YTD")

    try:
        sector_df, _ = load_sector_data()
        fig_sector = make_ytd_return_bar(sector_df, "sector", "return_ytd", "Sector Performance - YTD")
    except Exception:
        fig_sector = None
    add_plotly_figure_to_pdf_story(story, fig_sector, "Sector Performance - YTD")

    add_plotly_figure_to_pdf_story(story, make_market_yield_spread_fig(), "REIT Yield Spread - YTD")
    add_plotly_figure_to_pdf_story(story, make_sector_current_yield_fig(), "Current Dividend Yield by Sector")

    try:
        news_df, _ = load_sector_news_data()
        if not news_df.empty and "sector" in news_df.columns and "issue_tag" in news_df.columns:
            news_tmp = news_df.copy()
            news_tmp["issue_tag_split"] = news_tmp["issue_tag"].fillna("General").astype(str).str.split(", ")
            news_tmp = news_tmp.explode("issue_tag_split")
            issue_count = news_tmp.groupby(["sector", "issue_tag_split"]).size().reset_index(name="count").sort_values(["sector", "count"], ascending=[True, False])
            top_issue = issue_count.groupby("sector").head(1).reset_index(drop=True)
            rows = [["Sector", "Key Issue", "Count"]]
            for _, row in top_issue.iterrows():
                rows.append([str(row["sector"]), str(row["issue_tag_split"]), str(row["count"])])
            story.append(PageBreak())
            story.append(Paragraph("Sector News Issue Summary", styles["Heading2"]))
            tbl = Table(rows, colWidths=[1.6 * inch, 3.5 * inch, 0.8 * inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(tbl)
    except Exception:
        pass

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def render_download_report_panel():
    st.subheader("Download Dashboard Summary")
    st.caption("Download a text summary of the latest dashboard metrics. PDF export is intentionally removed to keep the app lightweight and deployment-safe.")
    today_str = pd.Timestamp.today().strftime("%Y%m%d")
    summary_text = build_market_summary_text()
    st.download_button(
        label="Download Summary TXT",
        data=summary_text,
        file_name=f"global_reit_dashboard_summary_{today_str}.txt",
        mime="text/plain",
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

                number_of_constituents = format_metric_value(
                    metric_map.get("number of constituents", {}).get("value"),
                    metric_map.get("number of constituents", {}).get("unit"),
                )

                net_market_cap = format_metric_value(
                    metric_map.get("net market cap", {}).get("value"),
                    metric_map.get("net market cap", {}).get("unit"),
                )

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
                            "ftse epra nareit developed",
                        )

                        daily_ytd_return = calc_ytd_return_from_timeseries(
                            macro_df,
                            date_col,
                            "ftse epra nareit developed",
                        )

                    if "dividend yield" in macro_df.columns:
                        # Use the same latest common date as Market Yield Spread
                        # so Overview Dividend Yield matches the valuation section.
                        market_yield_metrics = get_latest_market_yield_metrics(macro_df, date_col)
                        if market_yield_metrics is not None:
                            latest_dividend_yield = market_yield_metrics.get("dividend_yield")
                        else:
                            _, latest_dividend_yield = get_latest_from_timeseries(
                                macro_df,
                                date_col,
                                "dividend yield",
                            )

                daily_as_of_text = daily_as_of.strftime("%Y-%m-%d") if daily_as_of is not None else "-"
                daily_ytd_text = f"{daily_ytd_return:.2f}%" if daily_ytd_return is not None else "-"
                dividend_yield_text = f"{latest_dividend_yield:.2f}%" if latest_dividend_yield is not None else "-"

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
                    f"Daily Bloomberg index data as of {daily_as_of_text}. "
                    "Dividend Yield is sourced from Bloomberg Dividend 12Month Yield data and aligned with the Market Yield Spread section. "
                    "U.S. 10-year Treasury yield is collected via FRED DGS10 API. "
                    "Constituents, net market cap and country weight are factsheet-based as of 30 Apr 2026."
                )

                st.markdown('<div class="section-subtitle">Key Metrics</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c5, c6 = st.columns(2)
                c7, _ = st.columns(2)

                metric_cards = [
                    (c1, "As of Date", daily_as_of_text, "Daily Bloomberg data"),
                    (c2, "YTD Return", daily_ytd_text, "FTSE EPRA Nareit Developed"),
                    (c3, "Dividend Yield", dividend_yield_text, "Bloomberg index dividend yield"),
                    (c4, "Number of Constituents", number_of_constituents, "Factsheet as of 30 Apr 2026"),
                    (c5, "Best Region YTD", best_region_text, "Regional index performance"),
                    (c6, "Best Sector YTD", best_sector_text, "Sector index performance"),
                    (c7, "Net Market Cap", net_market_cap, "Factsheet as of 30 Apr 2026"),
                ]

                for col, label, value, note in metric_cards:
                    with col:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value}</div>
                                <div class="metric-note">{note}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
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
                    st.caption("Factsheet as of 30 Apr 2026. Labels show actual index weights, not re-normalized pie percentages.")

                    total_weight = plot_df[weight_col].sum()
                    has_us = plot_df[country_col].astype(str).str.lower().isin(["united states", "usa"]).any()
                    if total_weight < 95 or not has_us:
                        st.warning(
                            "Country weight data may be incomplete. Please rerun factsheet_download.py with USA alias handling. "
                            f"Current parsed total weight: {total_weight:.1f}%."
                        )

                    fig = px.pie(
                        plot_df,
                        names=country_col,
                        values=weight_col,
                        hole=0.55,
                        color_discrete_sequence=px.colors.sequential.Blues_r,
                    )
                    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{value:.1f}%")
                    fig.update_layout(
                        height=500,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("country / weight 컬럼을 찾지 못했습니다.")
        except Exception as e:
            st.warning("country_weight_latest.csv를 불러오지 못했습니다.")
            st.code(str(e))


# --------------------------------------------------
# Section: REIT vs Equity vs Rates
# --------------------------------------------------
def render_macro():
    st.header("REIT vs Equity vs Rates")
    st.caption(
        "Compares selected index returns and U.S. 10-year Treasury yields. "
        "Index returns are calculated from Bloomberg index levels, while UST 10Y is sourced from FRED DGS10 API."
    )

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
        max_n=3,
    )

    control_left, control_mid = st.columns([3, 1])

    with control_left:
        selected_index_cols = st.multiselect(
            "Select up to 3 index series",
            options=[c for c in numeric_cols if c != ust_col],
            default=index_default_cols[:3],
            max_selections=3,
            key="macro_indices",
        )

    with control_mid:
        period = st.selectbox(
            "Period",
            options=["YTD", "1Y", "3Y", "5Y", "10Y", "All"],
            index=0,
            key="macro_period",
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

    # Period performance summary cards
    st.subheader(f"{period} Performance Summary")
    kpi_cols = st.columns(len(selected_index_cols) + (1 if use_ust else 0))

    for i, col in enumerate(selected_index_cols):
        period_return = calc_period_return_from_timeseries(plot_df, date_col, col, period)
        value_text = "-" if period_return is None else f"{period_return:.2f}%"
        with kpi_cols[i]:
            st.metric(
                label=col,
                value=value_text,
                help=f"{period} return based on Bloomberg index levels",
            )

    if use_ust:
        latest_ust = get_latest_numeric_value(plot_df, date_col, ust_col)
        with kpi_cols[-1]:
            st.metric(
                label="UST 10Y",
                value="-" if latest_ust is None else f"{latest_ust:.2f}%",
                help="Latest U.S. 10-year Treasury yield from FRED DGS10 API",
            )

    line_df = plot_df[[date_col] + selected_index_cols].copy()

    for col in selected_index_cols:
        line_df[col] = to_base_100(line_df[col])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col in selected_index_cols:
        fig.add_trace(
            go.Scatter(
                x=line_df[date_col],
                y=line_df[col],
                mode="lines",
                name=col,
                line=dict(width=2.7),
            ),
            secondary_y=False,
        )

    if use_ust:
        ust_plot = plot_df[[date_col, ust_col]].dropna().copy()
        fig.add_trace(
            go.Bar(
                x=ust_plot[date_col],
                y=ust_plot[ust_col],
                name="UST 10Y (FRED DGS10)",
                opacity=0.35,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="REIT vs Equity vs Rates",
        height=580,
        hovermode="x unified",
        barmode="overlay",
        legend_title="Series",
        margin=dict(t=60, b=20, l=20, r=20),
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
        st.caption(f"Daily Bloomberg regional index data as of {as_of_date}")

    tab1, tab2, tab3 = st.tabs(["YTD", "1 Month", "Trailing 12 Months"])

    def draw_region_bar(data, value_col):
        plot_df = data.dropna(subset=[value_col]).sort_values(value_col, ascending=False)

        if plot_df.empty:
            st.info("해당 수익률 데이터가 없습니다.")
            return

        fig = px.bar(
            plot_df,
            x="region",
            y=value_col,
            color=value_col,
            color_continuous_scale="Blues",
            text=value_col,
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            height=500,
            xaxis_title="Region",
            yaxis_title="Return (%)",
            coloraxis_showscale=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab1:
        draw_region_bar(region_df, "return_ytd")

    with tab2:
        draw_region_bar(region_df, "return_1m")

    with tab3:
        draw_region_bar(region_df, "return_12m")


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
        st.caption(f"Daily Bloomberg sector index data as of {as_of_date}")

    tab1, tab2, tab3 = st.tabs(["YTD", "1 Month", "Trailing 12 Months"])

    def draw_sector_bar(data, value_col):
        plot_df = data.dropna(subset=[value_col]).sort_values(value_col, ascending=False)

        if plot_df.empty:
            st.info("해당 수익률 데이터가 없습니다.")
            return

        fig = px.bar(
            plot_df,
            x="sector",
            y=value_col,
            color=value_col,
            color_continuous_scale="Blues",
            text=value_col,
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            height=500,
            xaxis_title="Sector",
            yaxis_title="Return (%)",
            coloraxis_showscale=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab1:
        draw_sector_bar(sector_df, "return_ytd")

    with tab2:
        draw_sector_bar(sector_df, "return_1m")

    with tab3:
        draw_sector_bar(sector_df, "return_12m")


# --------------------------------------------------
# Subsection: Key Constituents Price Performance
# --------------------------------------------------
def render_key_constituents_price_performance():
    st.subheader("Key Constituents Price Performance")
    st.caption(
        "Bloomberg sector index returns are used for sector-level performance. "
        "All sector index members are available in the selection list, while the default chart uses the top constituents by index weight. "
        "Selected constituent price data is collected via yfinance API and converted to USD where required."
    )

    if yf is None:
        st.warning("yfinance 패키지가 설치되어 있지 않습니다. requirements.txt에 yfinance를 추가해주세요.")
        return

    try:
        ticker_df, _ = load_sector_ticker_universe()
    except Exception as e:
        st.warning("sector_ticker_universe.csv를 불러오지 못했습니다. 먼저 data_download.py를 실행해주세요.")
        st.code(str(e))
        return

    if ticker_df.empty:
        st.info("섹터 구성종목 데이터가 비어 있습니다.")
        return

    required_cols = ["sector", "bloomberg_ticker"]
    missing_cols = [c for c in required_cols if c not in ticker_df.columns]

    if missing_cols:
        st.error(f"sector_ticker_universe.csv에 필요한 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", ticker_df.columns.tolist())
        return

    ticker_df = ticker_df.copy()

    # The updated REITs Data.xlsx has INDX_MWEIGHT in Index Members_Value.
    # Use it as the primary representativeness measure. If unavailable, fall back to rank/source order.
    if "index_weight" in ticker_df.columns:
        ticker_df["index_weight"] = pd.to_numeric(ticker_df["index_weight"], errors="coerce")
    else:
        ticker_df["index_weight"] = pd.NA

    if "rank" in ticker_df.columns:
        ticker_df["rank"] = pd.to_numeric(ticker_df["rank"], errors="coerce")
    else:
        ticker_df["rank"] = pd.NA

    ticker_df["yf_ticker"] = ticker_df["bloomberg_ticker"].apply(bloomberg_to_yfinance_ticker)

    # Add listing metadata so users can see region/currency of each constituent.
    meta_df = ticker_df["bloomberg_ticker"].apply(get_bloomberg_listing_metadata).apply(pd.Series)
    ticker_df = pd.concat([ticker_df, meta_df], axis=1)

    sectors = sorted(ticker_df["sector"].dropna().unique().tolist())
    if not sectors:
        st.info("선택 가능한 섹터가 없습니다.")
        return

    c1, c2 = st.columns([2, 1])

    with c1:
        default_idx = sectors.index("Data Center") if "Data Center" in sectors else 0
        selected_sector = st.selectbox(
            "Select sector for constituent price check",
            options=sectors,
            index=default_idx,
            key="constituent_sector",
        )

    with c2:
        selected_period = st.selectbox(
            "Price history period",
            options=["YTD", "1mo", "3mo", "6mo", "1y", "3y"],
            index=0,
            key="constituent_period",
        )

    sector_tickers = ticker_df[ticker_df["sector"] == selected_sector].copy()

    # Sort by index weight first. This better reflects contribution to the sector index than simple file order.
    sector_tickers = sector_tickers.sort_values(
        by=["index_weight", "rank"],
        ascending=[False, True],
        na_position="last",
    )

    yf_tickers_all = (
        sector_tickers["yf_ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )
    yf_tickers_all = list(dict.fromkeys(yf_tickers_all))

    if not yf_tickers_all:
        st.info("이 섹터의 구성종목 중 yfinance에서 조회 가능한 ticker가 없습니다. 아래 mapping을 확인해주세요.")
        display_cols = [c for c in ["sector", "region", "country", "currency", "sector_index", "rank", "index_weight", "bloomberg_ticker", "yf_ticker"] if c in sector_tickers.columns]
        st.dataframe(sector_tickers[display_cols], use_container_width=True, hide_index=True)
        return

    name_map_all = get_yfinance_company_names(yf_tickers_all[:30])
    sector_tickers["company_name"] = sector_tickers["yf_ticker"].map(name_map_all)
    sector_tickers["display_name"] = sector_tickers.apply(
        lambda row: make_ticker_label(row.get("yf_ticker"), name_map_all),
        axis=1,
    )

    # Default chart tickers: top 5 by index weight. Users can add/remove tickers from the full constituent universe.
    default_tickers = yf_tickers_all[:5]
    option_labels = {}
    label_options = []

    for t in yf_tickers_all:
        row_match = sector_tickers[sector_tickers["yf_ticker"] == t]
        if not row_match.empty:
            r = row_match.iloc[0]
            company = r.get("company_name")
            weight = r.get("index_weight")
            if pd.notna(company) and str(company).strip() != "" and str(company) != t:
                base_label = f"{company} ({t})"
            else:
                base_label = str(t)
            if pd.notna(weight):
                label = f"{base_label} | Weight {float(weight):.2f}%"
            else:
                label = base_label
        else:
            label = str(t)
        option_labels[label] = t
        label_options.append(label)

    default_labels = [label for label, t in option_labels.items() if t in default_tickers]

    selected_labels = st.multiselect(
        "Select constituents to display in chart",
        options=label_options,
        default=default_labels,
        help="The full sector index member list is available here. The default selection uses the top 5 constituents by index weight.",
    )

    yf_tickers = [option_labels[label] for label in selected_labels]

    if not yf_tickers:
        st.info("차트에 표시할 종목을 최소 1개 선택해주세요.")
        return

    name_map = get_yfinance_company_names(yf_tickers)
    selected_df_for_caption = sector_tickers[sector_tickers["yf_ticker"].isin(yf_tickers)].copy()
    representative_text = ", ".join(
        selected_df_for_caption.apply(lambda row: make_ticker_label(row.get("yf_ticker"), name_map), axis=1).dropna().astype(str).tolist()
    )
    st.caption(f"Selected constituents: {representative_text}")
    st.caption("All available constituent price series are converted to USD for comparability. Returns are price returns and may differ from dividend-inclusive sector index returns. If yfinance has no usable data for a selected ticker, it remains in the table with a data-status flag.")

    price_df_local = load_yfinance_prices(yf_tickers, period=selected_period)

    if price_df_local.empty:
        st.info("yfinance에서 가격 데이터를 불러오지 못했습니다. 아래 ticker mapping을 확인해주세요.")
        display_cols = [c for c in ["sector", "region", "country", "currency", "sector_index", "rank", "index_weight", "bloomberg_ticker", "yf_ticker", "company_name"] if c in sector_tickers.columns]
        st.dataframe(sector_tickers[display_cols], use_container_width=True, hide_index=True)
        return

    price_df_local = price_df_local.ffill().dropna(how="all")
    valid_cols = [c for c in price_df_local.columns if price_df_local[c].dropna().shape[0] > 0]
    price_df_local = price_df_local[valid_cols]

    if price_df_local.empty:
        st.info("선택한 섹터의 구성종목 가격 데이터가 비어 있습니다.")
        return

    # Currency conversion to USD.
    currency_map = (
        sector_tickers.dropna(subset=["yf_ticker"])
        .drop_duplicates(subset=["yf_ticker"])
        .set_index("yf_ticker")["currency"]
        .to_dict()
    )
    price_df = convert_prices_to_usd(price_df_local, currency_map)
    price_df = price_df.ffill().dropna(how="all")

    normalized = price_df.copy()
    for col in normalized.columns:
        first_valid = normalized[col].dropna()
        if not first_valid.empty:
            normalized[col] = normalized[col] / first_valid.iloc[0] * 100

    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=make_ticker_label(col, name_map),
            )
        )

    fig.update_layout(
        title=f"Key Constituents Price Performance: {selected_sector} (USD Converted)",
        height=430,
        xaxis_title="Date",
        yaxis_title="USD price index (Start = 100)",
        hovermode="x unified",
        margin=dict(t=60, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Build a robust summary table. Include selected tickers even when yfinance has no usable prices.
    summary_rows = []
    available_tickers = list(price_df.columns)

    for ticker in yf_tickers:
        series = price_df[ticker].dropna() if ticker in price_df.columns else pd.Series(dtype="float64")
        if series.empty:
            latest_price = None
            ret = None
            status = "No yfinance price data"
        else:
            latest_price = series.iloc[-1]
            start_price = series.iloc[0]
            ret = None if pd.isna(start_price) or start_price == 0 else (latest_price / start_price - 1.0) * 100.0
            status = "OK"

        summary_rows.append(
            {
                "Ticker": ticker,
                "Latest Price (USD)": latest_price,
                f"Return ({selected_period}, USD)": ret,
                "Data Status": status,
            }
        )

    mapping_cols = ["bloomberg_ticker", "yf_ticker", "company_name", "region", "country", "currency", "index_weight", "rank"]
    mapping_cols = [c for c in mapping_cols if c in sector_tickers.columns]
    mapping_df = sector_tickers[mapping_cols].drop_duplicates(subset=["yf_ticker"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.merge(mapping_df, left_on="Ticker", right_on="yf_ticker", how="left")
    summary_df = summary_df.rename(
        columns={
            "company_name": "Company",
            "bloomberg_ticker": "Bloomberg Ticker",
            "region": "Region",
            "country": "Country",
            "currency": "Listing Currency",
            "index_weight": "Index Weight",
            "rank": "Rank",
        }
    )

    # Make missing company names readable.
    if "Company" in summary_df.columns:
        summary_df["Company"] = summary_df["Company"].fillna(summary_df["Ticker"])

    summary_df["Latest Price (USD)"] = summary_df["Latest Price (USD)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    summary_df[f"Return ({selected_period}, USD)"] = summary_df[f"Return ({selected_period}, USD)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")

    # Main table: hide Index Weight for readability. Show data status so missing names are not silently dropped.
    final_cols = [
        "Company",
        "Ticker",
        "Bloomberg Ticker",
        "Region",
        "Country",
        "Listing Currency",
        "Latest Price (USD)",
        f"Return ({selected_period}, USD)",
        "Data Status",
    ]
    final_cols = [c for c in final_cols if c in summary_df.columns]
    st.dataframe(summary_df[final_cols], use_container_width=True, hide_index=True)

    st.info(
        "Why sector index and constituent returns may differ: the Bloomberg sector index is a broad, weighted sector index, "
        "while this section shows selected constituents from the index universe. Constituent returns are USD-converted price returns from yfinance and do not include dividends; the sector index may reflect broader names, weights, FX and index methodology."
    )

    with st.expander(f"View all {selected_sector} index constituents"):
        full_df = sector_tickers.copy()
        display_cols = [c for c in ["sector", "region", "country", "currency", "sector_index", "rank", "index_weight", "bloomberg_ticker", "yf_ticker", "company_name"] if c in full_df.columns]
        if "index_weight" in full_df.columns:
            full_df["index_weight"] = full_df["index_weight"].map(lambda x: "-" if pd.isna(x) else f"{float(x):.2f}%")
        st.dataframe(full_df[display_cols], use_container_width=True, hide_index=True)



# --------------------------------------------------
# Section: REIT Dividend Yield Valuation
# --------------------------------------------------
def render_reit_dividend_yield_valuation():
    st.header("REIT Dividend Yield Valuation")
    st.caption(
        "Dividend yield is used as a valuation proxy. This section combines market-level REIT yield spread analysis "
        "and sector-level dividend yield valuation. UST 10Y source: FRED DGS10 API."
    )

    st.subheader("1. Market Yield Spread")

    try:
        macro_df, _ = load_macro_data()
    except Exception as e:
        st.warning("거시/배당수익률 데이터를 불러오지 못했습니다.")
        st.code(str(e))
        return

    date_col = pick_date_column(macro_df)
    if date_col is None or "dividend yield" not in macro_df.columns:
        st.warning("timeseries_macro.csv에서 date 또는 dividend yield 컬럼을 찾지 못했습니다.")
        st.write("현재 컬럼:", macro_df.columns.tolist())
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

    ust_col = guess_ust10y_col(numeric_cols)
    if ust_col is None:
        st.warning("UST 10Y 컬럼을 찾지 못했습니다.")
        return

    plot_df = macro_df[[date_col, "dividend yield", ust_col]].dropna().copy().sort_values(date_col)
    if plot_df.empty:
        st.info("시장 배당수익률/UST 10Y 데이터가 없습니다.")
        return

    plot_df["yield spread"] = plot_df["dividend yield"] - plot_df[ust_col]
    latest_date = plot_df[date_col].max()
    latest_row = plot_df.iloc[-1]
    latest_div_yield = latest_row["dividend yield"]
    latest_ust10y = latest_row[ust_col]
    latest_spread = latest_row["yield spread"]

    five_year_start = latest_date - pd.DateOffset(years=5)
    df_5y = plot_df[plot_df[date_col] >= five_year_start]
    avg_spread_5y = df_5y["yield spread"].mean() if not df_5y.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global REIT Dividend Yield", f"{latest_div_yield:.2f}%")
    c2.metric("UST 10Y (FRED)", f"{latest_ust10y:.2f}%")
    c3.metric("REIT Yield Spread", f"{latest_spread:.2f}%p")
    if avg_spread_5y is not None:
        c4.metric("5Y Avg Spread", f"{avg_spread_5y:.2f}%p")
    else:
        c4.metric("5Y Avg Spread", "-")

    period = st.selectbox(
        "Market yield spread period",
        options=["YTD", "1Y", "3Y", "5Y", "10Y", "All"],
        index=0,
        key="market_yield_period",
    )

    chart_df = filter_by_period(plot_df, date_col, period)

    col1, col2 = st.columns(2)
    with col1:
        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(x=chart_df[date_col], y=chart_df["dividend yield"], mode="lines", name="Global REIT Dividend Yield"))
        fig_yield.add_trace(go.Scatter(x=chart_df[date_col], y=chart_df[ust_col], mode="lines", name="UST 10Y (FRED DGS10)"))
        fig_yield.update_layout(
            title="Global REIT Dividend Yield vs UST 10Y",
            height=420,
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            hovermode="x unified",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with col2:
        fig_spread = px.line(chart_df, x=date_col, y="yield spread", title="REIT Yield Spread Trend")
        if avg_spread_5y is not None:
            fig_spread.add_hline(y=avg_spread_5y, line_dash="dash", annotation_text=f"5Y Avg: {avg_spread_5y:.2f}%p")
        fig_spread.add_hline(y=0, line_dash="dot", annotation_text="0%p")
        fig_spread.update_layout(
            height=420,
            xaxis_title="Date",
            yaxis_title="Spread (%p)",
            hovermode="x unified",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_spread, use_container_width=True)

    st.subheader("2. Sector Dividend Yield Valuation")

    try:
        yield_df, _ = load_sector_yield_data()
    except Exception as e:
        st.warning("섹터 dividend yield 데이터를 불러오지 못했습니다.")
        st.code(str(e))
        return

    y_date_col = pick_date_column(yield_df)
    if y_date_col is None:
        st.warning("섹터 yield 데이터에서 date 컬럼을 찾지 못했습니다.")
        st.write("현재 컬럼:", yield_df.columns.tolist())
        return

    yield_df = yield_df.copy()
    yield_df[y_date_col] = pd.to_datetime(yield_df[y_date_col], errors="coerce")
    yield_df = yield_df.dropna(subset=[y_date_col]).sort_values(y_date_col)

    sector_cols = [c for c in yield_df.columns if c != y_date_col]
    valid_sector_cols = []
    for c in sector_cols:
        yield_df[c] = pd.to_numeric(yield_df[c], errors="coerce")
        if yield_df[c].notna().sum() > 0:
            valid_sector_cols.append(c)
    sector_cols = valid_sector_cols

    if not sector_cols:
        st.info("섹터별 yield 숫자형 컬럼을 찾지 못했습니다.")
        return

    latest_date_sector = yield_df[y_date_col].max()
    cutoff_date = latest_date_sector - pd.DateOffset(years=5)
    yield_5y = yield_df[yield_df[y_date_col] >= cutoff_date].copy()

    latest_row_sector = yield_df.dropna(subset=sector_cols, how="all").iloc[-1]
    current = latest_row_sector[sector_cols]
    avg_5y = yield_5y[sector_cols].mean()

    summary = pd.DataFrame({
        "sector_raw": sector_cols,
        "sector": [clean_sector_yield_name(c) for c in sector_cols],
        "current_yield": current.values,
        "avg_5y_yield": avg_5y.values,
    })
    summary["yield_gap"] = summary["current_yield"] - summary["avg_5y_yield"]
    summary["us10y"] = latest_ust10y
    summary["spread_vs_us10y"] = summary["current_yield"] - latest_ust10y
    summary["view"] = summary.apply(classify_sector_valuation_view, axis=1)
    summary = summary.sort_values("current_yield", ascending=False)

    st.caption(f"Sector dividend yield data as of {latest_date_sector.strftime('%Y-%m-%d')}. 5Y averages and distributions use the last 5 years of observations.")

    highest_yield = summary.iloc[0]
    lowest_yield = summary.sort_values("current_yield", ascending=True).iloc[0]
    widest_gap = summary.sort_values("yield_gap", ascending=False).iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Current Yield", f"{highest_yield['sector']} ({highest_yield['current_yield']:.2f}%)")
    c2.metric("Lowest Current Yield", f"{lowest_yield['sector']} ({lowest_yield['current_yield']:.2f}%)")
    c3.metric("Largest Gap vs 5Y Avg", f"{widest_gap['sector']} ({widest_gap['yield_gap']:.2f}%p)")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_current = px.bar(summary, x="sector", y="current_yield", text="current_yield", title="Current Dividend Yield by Sector")
        fig_current.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_current.update_layout(height=430, xaxis_title="Sector", yaxis_title="Dividend Yield (%)", margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig_current, use_container_width=True)

    with col_b:
        fig_avg = go.Figure()
        fig_avg.add_bar(x=summary["sector"], y=summary["current_yield"], name="Current Yield")
        fig_avg.add_bar(x=summary["sector"], y=summary["avg_5y_yield"], name="5Y Average Yield")
        fig_avg.update_layout(barmode="group", title="Current Yield vs 5Y Average", height=430, xaxis_title="Sector", yaxis_title="Dividend Yield (%)", margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig_avg, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig_spread_sector = px.bar(summary, x="sector", y="spread_vs_us10y", text="spread_vs_us10y", title="Sector Yield Spread vs US10Y")
        fig_spread_sector.update_traces(texttemplate="%{text:.2f}%p", textposition="outside")
        fig_spread_sector.add_hline(y=0, line_dash="dash")
        fig_spread_sector.update_layout(height=430, xaxis_title="Sector", yaxis_title="Spread vs US10Y (%p)", margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig_spread_sector, use_container_width=True)

    with col_d:
        long_df = yield_5y[[y_date_col] + sector_cols].copy()
        long_df = long_df.melt(id_vars=y_date_col, value_vars=sector_cols, var_name="sector_raw", value_name="dividend_yield")
        long_df["sector"] = long_df["sector_raw"].apply(clean_sector_yield_name)
        long_df = long_df.dropna(subset=["dividend_yield"])
        fig_box = px.box(long_df, x="sector", y="dividend_yield", title="5Y Historical Dividend Yield Distribution")
        fig_box.add_scatter(x=summary["sector"], y=summary["current_yield"], mode="markers", name="Current Yield", marker=dict(size=9))
        fig_box.add_annotation(
            text="Box plot based on the last 5 years; marker indicates current yield.",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            font=dict(size=11, color="gray"), align="left",
        )
        fig_box.update_layout(height=430, xaxis_title="Sector", yaxis_title="Dividend Yield (%)", margin=dict(t=70, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Sector Valuation Summary")
    if summary["us10y"].notna().any():
        us10y_ref = summary["us10y"].dropna().iloc[0]
        st.caption(f"US10Y reference rate: {us10y_ref:.2f}% from FRED DGS10 API")

    summary_for_display = summary.copy()
    summary_for_display["Sector"] = summary_for_display["sector"]
    summary_for_display["Current Yield"] = summary_for_display["current_yield"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
    summary_for_display["5Y Avg Yield"] = summary_for_display["avg_5y_yield"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
    summary_for_display["Gap vs 5Y Avg"] = summary_for_display["yield_gap"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%p")
    summary_for_display["Spread vs US10Y"] = summary_for_display["spread_vs_us10y"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%p")
    summary_for_display["View"] = summary_for_display["view"]
    summary_for_display = summary_for_display[["Sector", "Current Yield", "5Y Avg Yield", "Gap vs 5Y Avg", "Spread vs US10Y", "View"]]

    def color_view_column(value):
        value = str(value)
        if "Relatively cheap" in value:
            return "background-color: #14532d; color: #dcfce7; font-weight: 700;"
        if "Cheap vs history" in value:
            return "background-color: #713f12; color: #fef3c7; font-weight: 700;"
        if "Premium" in value:
            return "background-color: #7f1d1d; color: #fee2e2; font-weight: 700;"
        if "Neutral" in value:
            return "background-color: #374151; color: #e5e7eb; font-weight: 700;"
        return "background-color: #1f2937; color: #e5e7eb;"

    try:
        styled_summary = summary_for_display.style.map(color_view_column, subset=["View"])
    except AttributeError:
        styled_summary = summary_for_display.style.applymap(color_view_column, subset=["View"])
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    st.info(
        "Interpretation: Higher current dividend yield versus the 5-year average may indicate cheaper valuation, "
        "while spread over US10Y shows bond-relative income appeal. Premium sectors may reflect stronger growth expectations or lower perceived risk."
    )


# --------------------------------------------------
# Section: Sector News Monitor
# --------------------------------------------------
def render_sector_news():
    st.header("Sector News Monitor")
    st.caption(
        "Google News RSS headlines are scraped using sector keywords and selected index member tickers. "
        "This section summarizes scraped news by sector and issue tag, then provides underlying article cards for review."
    )

    try:
        news_df, _ = load_sector_news_data()
    except Exception as e:
        st.warning("sector_news.csv를 불러오지 못했습니다. 먼저 news_download.py를 실행해주세요.")
        st.code(str(e))
        return

    if news_df.empty:
        st.info("뉴스 데이터가 비어 있습니다.")
        return

    required_cols = ["sector", "title", "source", "published", "summary", "link", "issue_tag"]
    missing_cols = [c for c in required_cols if c not in news_df.columns]

    if missing_cols:
        st.error(f"sector_news.csv에 필요한 컬럼이 없습니다: {missing_cols}")
        st.write("현재 컬럼:", news_df.columns.tolist())
        return

    news_df = news_df.copy()
    if "published_datetime" in news_df.columns:
        news_df["published_datetime"] = pd.to_datetime(news_df["published_datetime"], errors="coerce", utc=True)
    else:
        news_df["published_datetime"] = pd.to_datetime(news_df["published"], errors="coerce", utc=True)

    news_df["issue_tag"] = news_df["issue_tag"].fillna("General").astype(str)
    news_df["sector"] = news_df["sector"].fillna("Unknown").astype(str)
    news_df = news_df.sort_values("published_datetime", ascending=False)

    # --------------------------------------------------
    # News collection metadata KPIs
    # --------------------------------------------------
    total_articles = len(news_df)
    total_queries = news_df["query"].dropna().nunique() if "query" in news_df.columns else "-"
    most_active_sector = news_df["sector"].value_counts().index[0] if total_articles > 0 else "-"

    tag_series = news_df["issue_tag"].str.split(", ").explode().replace("", pd.NA).dropna()
    non_general_tag_series = tag_series[tag_series != "General"]
    if not non_general_tag_series.empty:
        most_mentioned_issue = non_general_tag_series.value_counts().index[0]
    elif not tag_series.empty:
        most_mentioned_issue = tag_series.value_counts().index[0]
    else:
        most_mentioned_issue = "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Articles", f"{total_articles}")
    c2.metric("Lookback Period", "Last 30 Days")
    c3.metric("Search Queries", total_queries)
    c4.metric("Most Active Sector", most_active_sector)

    st.caption(
        f"Most mentioned non-general issue tag: **{most_mentioned_issue}**. "
        "Search queries combine sector keywords and selected index member tickers."
    )

    # --------------------------------------------------
    # Issue summary by sector
    # --------------------------------------------------
    st.subheader("Sector Issue Summary")
    st.caption(
        "This table is the main scraping output: it summarizes which issues are most visible in the collected headlines by sector."
    )

    issue_long = news_df.copy()
    issue_long["issue_tag_split"] = issue_long["issue_tag"].str.split(", ")
    issue_long = issue_long.explode("issue_tag_split")
    issue_long["issue_tag_split"] = issue_long["issue_tag_split"].fillna("General")

    issue_count_df = (
        issue_long
        .groupby(["sector", "issue_tag_split"])
        .size()
        .reset_index(name="count")
        .sort_values(["sector", "count"], ascending=[True, False])
    )

    def top_issue_text(sector_name, n=3):
        temp = issue_count_df[issue_count_df["sector"] == sector_name].copy()
        temp_non_general = temp[temp["issue_tag_split"] != "General"]
        if not temp_non_general.empty:
            temp = temp_non_general
        temp = temp.head(n)
        if temp.empty:
            return "No clear issue tag"
        return "; ".join([f"{r['issue_tag_split']} ({int(r['count'])})" for _, r in temp.iterrows()])

    def latest_date_text(sector_name):
        temp = news_df[news_df["sector"] == sector_name].copy().dropna(subset=["published_datetime"])
        if temp.empty:
            return "-"
        return temp["published_datetime"].max().strftime("%Y-%m-%d")

    def top_sources_text(sector_name, n=2):
        temp = news_df[news_df["sector"] == sector_name].copy()
        if "source" not in temp.columns or temp.empty:
            return "-"
        counts = temp["source"].fillna("Unknown").replace("", "Unknown").value_counts().head(n)
        if counts.empty:
            return "-"
        return "; ".join([f"{idx} ({val})" for idx, val in counts.items()])

    sector_rows = []
    for sector_name in sorted(news_df["sector"].dropna().unique()):
        sector_articles = news_df[news_df["sector"] == sector_name]
        sector_rows.append({
            "Sector": sector_name,
            "Articles": len(sector_articles),
            "Top Issues": top_issue_text(sector_name, n=3),
            "Latest News Date": latest_date_text(sector_name),
            "Top Sources": top_sources_text(sector_name, n=2),
        })

    sector_summary_df = pd.DataFrame(sector_rows).sort_values("Articles", ascending=False)
    st.dataframe(sector_summary_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # News takeaways: interpretation separated from raw summary table
    # --------------------------------------------------
    st.subheader("News Takeaways")
    st.caption(
        "These are rule-based takeaways from scraped headlines and issue tags. "
        "They are designed to summarize the news signal without repeating the full article list below."
    )

    non_general_issue_count = (
        issue_long[issue_long["issue_tag_split"] != "General"]
        .groupby("issue_tag_split")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    def issue_count_for_sector(sector_name, issue_name):
        temp = issue_count_df[
            (issue_count_df["sector"] == sector_name)
            & (issue_count_df["issue_tag_split"] == issue_name)
        ]
        if temp.empty:
            return 0
        return int(temp["count"].iloc[0])

    def sector_exists(name_part):
        temp = sector_summary_df[
            sector_summary_df["Sector"].astype(str).str.lower().str.contains(name_part, na=False)
        ]
        return None if temp.empty else temp.iloc[0]["Sector"]

    takeaway_lines = []

    if not non_general_issue_count.empty:
        top_issue = non_general_issue_count.iloc[0]["issue_tag_split"]
        top_count = int(non_general_issue_count.iloc[0]["count"])
        takeaway_lines.append(
            f"**{top_issue}** is the most visible issue across the scraped REIT news universe, "
            f"appearing in **{top_count}** tagged articles."
        )

    data_center_sector = sector_exists("data center")
    if data_center_sector is not None:
        ai_count = issue_count_for_sector(data_center_sector, "AI / Data Center Demand")
        ma_count = issue_count_for_sector(data_center_sector, "M&A / Transaction")
        earn_count = issue_count_for_sector(data_center_sector, "Earnings / Guidance")
        details = []
        if ai_count:
            details.append(f"AI/data-center demand ({ai_count})")
        if ma_count:
            details.append(f"transaction activity ({ma_count})")
        if earn_count:
            details.append(f"earnings/guidance ({earn_count})")
        detail_text = ", ".join(details) if details else "data-center related headlines"
        takeaway_lines.append(
            f"**Data Center** news flow is concentrated in **{detail_text}**, which is consistent with the sector being driven by digital infrastructure demand, cloud capacity, and power availability themes."
        )

    office_sector = sector_exists("office")
    # Avoid using Industrial / Office as pure Office if pure Office exists
    if office_sector is not None:
        office_vacancy_count = issue_count_for_sector(office_sector, "Office Vacancy Risk")
        rates_count = issue_count_for_sector(office_sector, "Rates / Treasury Yield")
        ma_count = issue_count_for_sector(office_sector, "M&A / Transaction")
        if office_vacancy_count or rates_count or ma_count:
            takeaway_lines.append(
                f"**Office** headlines remain more relevant when read through vacancy/leasing risk "
                f"({office_vacancy_count}), rate sensitivity ({rates_count}), and transaction/refinancing themes ({ma_count}). "
                "This helps distinguish Office from growth-led sectors such as Data Center."
            )

    retail_sector = sector_exists("retail")
    if retail_sector is not None:
        retail_count = issue_count_for_sector(retail_sector, "Retail / Consumer Demand")
        earnings_count = issue_count_for_sector(retail_sector, "Earnings / Guidance")
        takeaway_lines.append(
            f"**Retail** should be interpreted through consumer demand, tenant sales, and earnings flow. "
            f"The scraped tags show retail/consumer demand ({retail_count}) and earnings/guidance ({earnings_count}); individual constituent returns can still diverge from the sector index because of weights, geography, FX and dividend treatment."
        )

    healthcare_sector = sector_exists("healthcare")
    if healthcare_sector is not None:
        senior_count = issue_count_for_sector(healthcare_sector, "Healthcare / Senior Housing")
        housing_count = issue_count_for_sector(healthcare_sector, "Housing / Rent Growth")
        earnings_count = issue_count_for_sector(healthcare_sector, "Earnings / Guidance")
        takeaway_lines.append(
            f"**Healthcare** has a more sector-specific news signal: senior housing/healthcare ({senior_count}), housing/rent growth ({housing_count}), and earnings/guidance ({earnings_count}) appear in the scraped headlines."
        )

    if not takeaway_lines:
        takeaway_lines.append(
            "No strong non-general issue pattern was detected. Consider expanding issue keyword rules in news_download.py."
        )

    for i, line in enumerate(takeaway_lines[:5], start=1):
        st.markdown(f"{i}. {line}")

    # --------------------------------------------------
    # Issue distribution chart
    # --------------------------------------------------
    st.subheader("Issue Tag Distribution")
    issue_chart_df = (
        issue_long[issue_long["issue_tag_split"] != "General"]
        .groupby("issue_tag_split")
        .size()
        .reset_index(name="articles")
        .sort_values("articles", ascending=True)
    )

    if issue_chart_df.empty:
        st.info("General 이외의 issue tag가 충분하지 않습니다. news_download.py의 issue keyword rules를 보완하면 더 잘 나옵니다.")
    else:
        fig_issue = px.bar(
            issue_chart_df,
            x="articles",
            y="issue_tag_split",
            orientation="h",
            text="articles",
            title="Scraped News Count by Issue Tag",
        )
        fig_issue.update_traces(textposition="outside")
        fig_issue.update_layout(
            height=430,
            xaxis_title="Number of articles",
            yaxis_title="Issue tag",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_issue, use_container_width=True)

    # --------------------------------------------------
    # Focused article review
    # --------------------------------------------------
    st.subheader("Latest Articles")
    st.caption("Use this as source-level evidence behind the sector issue summary above.")

    sectors = sorted(news_df["sector"].dropna().unique().tolist())
    selected_sector = st.selectbox("Select Sector", options=["All"] + sectors, index=0, key="news_sector")

    available_issues = sorted(issue_long["issue_tag_split"].dropna().unique().tolist())
    selected_issue = st.selectbox("Filter by Issue Tag", options=["All"] + available_issues, index=0, key="news_issue_filter")

    if selected_sector != "All":
        display_df = news_df[news_df["sector"] == selected_sector].copy()
    else:
        display_df = news_df.copy()

    if selected_issue != "All":
        display_df = display_df[display_df["issue_tag"].astype(str).str.contains(selected_issue, regex=False, na=False)].copy()

    max_articles = st.slider("Number of articles to show", min_value=3, max_value=20, value=6, step=1, key="news_articles")
    display_df = display_df.head(max_articles)

    if display_df.empty:
        st.info("선택한 필터에 해당하는 기사가 없습니다.")
    else:
        for _, row in display_df.iterrows():
            sector = row.get("sector", "")
            title = row.get("title", "")
            source = row.get("source", "")
            published = row.get("published", "")
            summary = row.get("summary", "")
            link = row.get("link", "")
            issue_tag = row.get("issue_tag", "General")
            query = row.get("query", "")
            query_type = row.get("query_type", "")
            source_ticker = row.get("source_ticker", "")

            if pd.isna(title) or str(title).strip() == "":
                continue

            source_line = f"Source Query: {query}"
            if pd.notna(source_ticker) and str(source_ticker).strip() != "":
                source_line += f" | Source Ticker: {source_ticker}"
            else:
                source_line += f" | Source Type: {query_type}"

            st.markdown(
                f'''
                <div style="
                    border: 1px solid #E2E8F0;
                    border-radius: 12px;
                    padding: 14px 16px;
                    margin-bottom: 12px;
                    background-color: #FFFFFF;
                    box-shadow: 0 1px 2px rgba(15,23,42,0.08);
                ">
                    <div style="font-size:0.80rem; color:#64748B; margin-bottom:6px;">
                        {sector} | {source} | {published}
                    </div>
                    <div style="font-size:0.78rem; color:#2563EB; margin-bottom:6px; font-weight:700;">
                        Issue Tag: {issue_tag}
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; margin-bottom:8px; color:#0F172A;">
                        <a href="{link}" target="_blank" style="text-decoration:none; color:#1D4ED8;">
                            {title}
                        </a>
                    </div>
                    <div style="font-size:0.88rem; color:#334155; margin-bottom:8px;">
                        {summary}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B;">
                        {source_line}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    with st.expander("View tickers used for scraping"):
        try:
            ticker_df, _ = load_sector_ticker_universe()
            display_cols = [c for c in ["sector", "sector_index", "rank", "index_weight", "bloomberg_ticker", "search_ticker"] if c in ticker_df.columns]
            sort_cols = [c for c in ["sector", "rank"] if c in ticker_df.columns]
            if "index_weight" in ticker_df.columns:
                ticker_df["index_weight"] = pd.to_numeric(ticker_df["index_weight"], errors="coerce")
            st.dataframe(ticker_df[display_cols].sort_values(sort_cols) if sort_cols else ticker_df[display_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.info("sector_ticker_universe.csv 파일이 없거나 불러올 수 없습니다.")
            st.code(str(e))


# --------------------------------------------------
# Section: Sector Performance & News Link
# --------------------------------------------------
def normalize_sector_name_for_join(name):
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = text.replace("centers", "center")
    text = text.replace("self-storage", "self storage")
    text = text.replace("selfstorage", "self storage")
    text = text.replace("lodgings", "lodging")
    text = text.replace("resorts", "resort")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


SECTOR_PRIORITY_ISSUES = {
    "data center": [
        "AI / Data Center Demand",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
    "office": [
        "Office Vacancy Risk",
        "Rates / Treasury Yield",
        "Earnings / Guidance",
        "M&A / Transaction",
    ],
    "retail": [
        "Retail / Consumer Demand",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
    "healthcare": [
        "Healthcare / Senior Housing",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
    "industrial": [
        "Industrial / Logistics Demand",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
    "residential": [
        "Housing / Rent Growth",
        "Rates / Treasury Yield",
        "Earnings / Guidance",
        "M&A / Transaction",
    ],
    "self storage": [
        "Self Storage Demand",
        "Earnings / Guidance",
        "Rates / Treasury Yield",
    ],
    "lodging resort": [
        "Hotels / Travel Demand",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
    "specialty": [
        "Net Lease / Tenant Credit",
        "Earnings / Guidance",
        "M&A / Transaction",
        "Rates / Treasury Yield",
    ],
}


def build_issue_phrase(issue_rows):
    if issue_rows.empty:
        return "No dominant non-general issue tag was detected."
    parts = []
    for _, row in issue_rows.iterrows():
        parts.append(f"{row['issue_tag_split']} ({int(row['article_count'])})")
    return "; ".join(parts)


def get_relevant_issue_rows(news_df, sector_key, max_issues=3):
    temp = news_df[news_df["sector_key"] == sector_key].copy()
    if temp.empty:
        return pd.DataFrame(columns=["issue_tag_split", "article_count"])

    temp["issue_tag"] = temp["issue_tag"].fillna("General").astype(str)
    temp["issue_tag_split"] = temp["issue_tag"].str.split(", ")
    long_df = temp.explode("issue_tag_split")
    long_df["issue_tag_split"] = long_df["issue_tag_split"].fillna("General")
    long_df = long_df[long_df["issue_tag_split"] != "General"]

    if long_df.empty:
        return pd.DataFrame(columns=["issue_tag_split", "article_count"])

    issue_count = (
        long_df.groupby("issue_tag_split")
        .size()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
    )

    priority = SECTOR_PRIORITY_ISSUES.get(sector_key, [])
    if priority:
        issue_count["priority_rank"] = issue_count["issue_tag_split"].apply(
            lambda x: priority.index(x) if x in priority else 999
        )
        # 우선순위가 있는 섹터 특화 이슈를 먼저 보여주고, 같은 우선순위에서는 기사 수가 많은 순서
        issue_count = issue_count.sort_values(["priority_rank", "article_count"], ascending=[True, False])

    return issue_count.head(max_issues)


def get_related_headlines(news_df, sector_key, issue_rows, max_headlines=3):
    temp = news_df[news_df["sector_key"] == sector_key].copy()
    if temp.empty:
        return []

    issue_set = set(issue_rows["issue_tag_split"].dropna().astype(str).tolist())
    if issue_set:
        mask = temp["issue_tag"].fillna("").astype(str).apply(lambda x: any(issue in x for issue in issue_set))
        filtered = temp[mask].copy()
        if not filtered.empty:
            temp = filtered

    if "published_datetime" in temp.columns:
        temp["published_datetime"] = pd.to_datetime(temp["published_datetime"], errors="coerce")
        temp = temp.sort_values("published_datetime", ascending=False)

    headlines = []
    for _, row in temp.head(max_headlines).iterrows():
        title = str(row.get("title", "")).strip()
        source = str(row.get("source", "")).strip()
        if title:
            headlines.append(f"{title} ({source})" if source else title)
    return headlines


def get_constituent_performance_snapshot(sector_name, period_label):
    """Return a short USD-return snapshot for top index constituents in a sector."""
    if yf is None:
        return {"available": False, "text": "Constituent price data is unavailable because yfinance is not installed.", "rows": pd.DataFrame()}

    try:
        ticker_df, _ = load_sector_ticker_universe()
    except Exception:
        return {"available": False, "text": "Constituent universe file is unavailable.", "rows": pd.DataFrame()}

    if ticker_df.empty or "sector" not in ticker_df.columns or "bloomberg_ticker" not in ticker_df.columns:
        return {"available": False, "text": "Constituent universe data is unavailable.", "rows": pd.DataFrame()}

    ticker_df = ticker_df.copy()
    ticker_df["sector_key"] = ticker_df["sector"].apply(normalize_sector_name_for_join)
    sector_key = normalize_sector_name_for_join(sector_name)
    sector_tickers = ticker_df[ticker_df["sector_key"] == sector_key].copy()

    if sector_tickers.empty:
        return {"available": False, "text": "No constituents were matched for this sector.", "rows": pd.DataFrame()}

    if "index_weight" in sector_tickers.columns:
        sector_tickers["index_weight"] = pd.to_numeric(sector_tickers["index_weight"], errors="coerce")
    else:
        sector_tickers["index_weight"] = pd.NA

    if "rank" in sector_tickers.columns:
        sector_tickers["rank"] = pd.to_numeric(sector_tickers["rank"], errors="coerce")
    else:
        sector_tickers["rank"] = pd.NA

    sector_tickers["yf_ticker"] = sector_tickers["bloomberg_ticker"].apply(bloomberg_to_yfinance_ticker)
    meta_df = sector_tickers["bloomberg_ticker"].apply(get_bloomberg_listing_metadata).apply(pd.Series)
    sector_tickers = pd.concat([sector_tickers.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    sector_tickers = sector_tickers.sort_values(["index_weight", "rank"], ascending=[False, True], na_position="last")

    yf_tickers = (
        sector_tickers["yf_ticker"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().tolist()
    )
    yf_tickers = list(dict.fromkeys(yf_tickers))[:5]

    if not yf_tickers:
        return {"available": False, "text": "No yfinance-compatible tickers were available for this sector.", "rows": pd.DataFrame()}

    price_df = load_yfinance_prices(yf_tickers, period=period_label)
    if price_df.empty:
        return {"available": False, "text": "yfinance returned no price data for the selected constituents.", "rows": pd.DataFrame()}

    currency_map = {}
    for t in yf_tickers:
        matched = sector_tickers[sector_tickers["yf_ticker"] == t]
        if not matched.empty:
            currency_map[t] = matched.iloc[0].get("currency", "USD")
        else:
            currency_map[t] = "USD"

    price_usd = convert_prices_to_usd(price_df, currency_map).ffill().dropna(how="all")
    valid_cols = [c for c in price_usd.columns if price_usd[c].dropna().shape[0] >= 2]
    price_usd = price_usd[valid_cols]

    if price_usd.empty:
        return {"available": False, "text": "No valid USD-converted price series were available.", "rows": pd.DataFrame()}

    returns = (price_usd.ffill().iloc[-1] / price_usd.ffill().iloc[0] - 1.0) * 100.0
    name_map = get_yfinance_company_names(list(returns.index))

    rows = []
    for ticker, ret in returns.items():
        matched = sector_tickers[sector_tickers["yf_ticker"] == ticker]
        bbg = matched.iloc[0].get("bloomberg_ticker", "") if not matched.empty else ""
        country = matched.iloc[0].get("country", "") if not matched.empty else ""
        company = name_map.get(ticker, ticker)
        rows.append({
            "Company": company,
            "Ticker": ticker,
            "Bloomberg Ticker": bbg,
            "Country": country,
            "Return": ret,
        })

    out = pd.DataFrame(rows).sort_values("Return", ascending=False)
    if out.empty:
        return {"available": False, "text": "No constituent return data was available.", "rows": pd.DataFrame()}

    top = out.iloc[0]
    bottom = out.iloc[-1]
    positive_count = int((out["Return"] > 0).sum())
    negative_count = int((out["Return"] < 0).sum())

    text = (
        f"Among the selected top constituents, {positive_count} were positive and {negative_count} were negative over {period_label}. "
        f"The strongest selected name was {top['Company']} ({top['Ticker']}, {top['Return']:.2f}%), "
        f"while the weakest was {bottom['Company']} ({bottom['Ticker']}, {bottom['Return']:.2f}%)."
    )

    return {"available": True, "text": text, "rows": out}


def build_deep_sector_comment(sector_name, sector_key, return_value, period_label, issue_rows, constituent_snapshot, role):
    issue_names = issue_rows["issue_tag_split"].dropna().astype(str).tolist() if not issue_rows.empty else []
    issue_text = build_issue_phrase(issue_rows)
    ret_text = f"{return_value:.2f}%"

    direction_word = "outperformance" if return_value >= 0 else "underperformance"
    if role == "best":
        opening = f"{sector_name} was the best-performing sector over {period_label}, with a {ret_text} return."
    else:
        opening = f"{sector_name} was the weakest sector over {period_label}, with a {ret_text} return."

    # Sector-specific analytical interpretation
    if sector_key == "data center":
        reason = (
            "The scraped issue mix points to AI-driven data center demand, cloud expansion, and power-capacity constraints. "
            "These are structural growth themes, so the sector's performance is better interpreted as demand-led digital infrastructure repricing rather than only a broad REIT beta move."
        )
    elif sector_key == "office":
        reason = (
            "The relevant office issues are vacancy risk, weak leasing demand, refinancing pressure, and rate sensitivity. "
            "If Office is the weakest sector, the news flow is consistent with structural demand uncertainty and balance-sheet pressure rather than a simple cyclical pullback."
        )
    elif sector_key == "retail":
        reason = (
            "Retail should be read through consumer spending, tenant sales, shopping-center traffic, and earnings momentum. "
            "If the sector index is strong while selected constituents are weaker, the gap can reflect index weights, geography, FX, and the difference between a broad Bloomberg index and selected yfinance price returns."
        )
    elif sector_key == "healthcare":
        reason = (
            "Healthcare REIT performance is usually tied to senior-housing occupancy, rent growth, operating recovery, and earnings guidance. "
            "Positive senior-housing or earnings news can support the sector, while higher rates can still pressure valuation multiples."
        )
    elif sector_key == "industrial":
        reason = (
            "Industrial REIT performance is linked to warehouse leasing, logistics demand, supply-chain activity, and e-commerce-related space demand. "
            "A strong issue signal around logistics or leasing supports the performance narrative; weaker leasing or oversupply headlines would weaken it."
        )
    elif sector_key == "residential":
        reason = (
            "Residential REITs are sensitive to rent growth, apartment demand, housing affordability, and interest rates. "
            "The sector's performance should therefore be interpreted through both rental fundamentals and rate-driven valuation pressure."
        )
    elif sector_key == "lodging resort":
        reason = (
            "Lodging and resort REITs are closely linked to travel demand, occupancy, RevPAR, and earnings momentum. "
            "A strong performance period is more credible when travel-demand or earnings tags appear in the scraped news."
        )
    elif sector_key == "self storage":
        reason = (
            "Self-storage performance is usually driven by move-in activity, occupancy, rental demand, and operating trends. "
            "News flow around demand normalization or earnings can explain why the sector diverges from broader REITs."
        )
    elif sector_key == "specialty":
        reason = (
            "Specialty REITs are a mixed group, so the interpretation depends on the underlying issue mix, including net lease, gaming, tenant credit, transactions, and earnings. "
            "Because the sector is heterogeneous, constituent-level evidence is especially important."
        )
    else:
        reason = (
            "The scraped news provides a qualitative signal that should be read together with sector return, valuation, and constituent performance."
        )

    transaction_note = ""
    if any("M&A / Transaction" in x for x in issue_names):
        transaction_note = " Transaction-related news, including M&A, IPO, or portfolio activity, was also detected, which can create stock-specific moves or sector re-rating noise."

    earnings_note = ""
    if any("Earnings / Guidance" in x for x in issue_names):
        earnings_note = " Earnings or guidance tags were present, so part of the move may be linked to company fundamentals rather than only macro or sector beta."

    constituent_text = constituent_snapshot.get("text", "") if constituent_snapshot else ""

    comment = (
        f"{opening} Key scraped issues were: {issue_text}. {reason}"
        f"{earnings_note}{transaction_note}"
    )

    if constituent_text:
        comment += f" Constituent evidence: {constituent_text}"

    return comment


def render_sector_performance_news_link():
    st.header("Sector Issue Brief")
    st.caption(
        "This section connects sector performance with sector-specific scraped news issues and selected constituent returns. "
        "It is a qualitative issue brief, not a causal model."
    )

    try:
        sector_df, _ = load_sector_data()
        news_df, _ = load_sector_news_data()
    except Exception as e:
        st.info("섹터 성과 또는 뉴스 데이터를 불러오지 못했습니다. data_download.py와 news_download.py 실행 여부를 확인해주세요.")
        st.code(str(e))
        return

    if sector_df.empty or news_df.empty:
        st.info("섹터 성과 또는 뉴스 데이터가 비어 있습니다.")
        return

    period_options = {
        "YTD": "return_ytd",
        "1 Month": "return_1m",
        "Trailing 12 Months": "return_12m",
    }

    c_period, _ = st.columns([1, 3])
    with c_period:
        selected_period_label = st.selectbox(
            "Issue brief performance period",
            options=list(period_options.keys()),
            index=0,
            help="Best and weakest sectors are selected based on this period. YTD is the default for the project narrative.",
        )

    return_col = period_options[selected_period_label]

    if "sector" not in sector_df.columns or return_col not in sector_df.columns:
        st.info(f"sector_returns.csv에 sector / {return_col} 컬럼이 필요합니다.")
        return

    required_news_cols = ["sector", "issue_tag", "title", "source", "published"]
    missing_news_cols = [c for c in required_news_cols if c not in news_df.columns]
    if missing_news_cols:
        st.info(f"sector_news.csv에 필요한 컬럼이 없습니다: {missing_news_cols}")
        return

    sector_df = sector_df.copy()
    news_df = news_df.copy()
    sector_df["sector_key"] = sector_df["sector"].apply(normalize_sector_name_for_join)
    news_df["sector_key"] = news_df["sector"].apply(normalize_sector_name_for_join)
    sector_df[return_col] = pd.to_numeric(sector_df[return_col], errors="coerce")
    sector_df = sector_df.dropna(subset=["sector", return_col, "sector_key"])

    if sector_df.empty:
        st.info(f"{selected_period_label} 섹터 수익률 데이터가 없습니다.")
        return

    best_sector_row = sector_df.sort_values(return_col, ascending=False).iloc[0]
    weakest_sector_row = sector_df.sort_values(return_col, ascending=True).iloc[0]

    best_issues = get_relevant_issue_rows(news_df, best_sector_row["sector_key"], max_issues=3)
    weak_issues = get_relevant_issue_rows(news_df, weakest_sector_row["sector_key"], max_issues=3)

    # Constituent performance is aligned with the same period as the issue brief where possible.
    yf_period_map = {
        "YTD": "YTD",
        "1 Month": "1mo",
        "Trailing 12 Months": "1y",
    }
    constituent_period = yf_period_map.get(selected_period_label, "YTD")
    best_constituents = get_constituent_performance_snapshot(best_sector_row["sector"], constituent_period)
    weak_constituents = get_constituent_performance_snapshot(weakest_sector_row["sector"], constituent_period)

    best_comment = build_deep_sector_comment(
        best_sector_row["sector"],
        best_sector_row["sector_key"],
        best_sector_row[return_col],
        selected_period_label,
        best_issues,
        best_constituents,
        role="best",
    )
    weak_comment = build_deep_sector_comment(
        weakest_sector_row["sector"],
        weakest_sector_row["sector_key"],
        weakest_sector_row[return_col],
        selected_period_label,
        weak_issues,
        weak_constituents,
        role="weakest",
    )

    def render_issue_card(title, row, issues, comment, constituents, role_color):
        st.markdown(f"### {title}")
        st.metric(
            label=f"Sector / {selected_period_label} Return",
            value=row["sector"],
            delta=f"{row[return_col]:.2f}%",
        )

        st.markdown("**Key scraped issue signals**")
        if issues.empty:
            st.markdown("- No dominant non-general issue tag detected")
        else:
            for _, issue_row in issues.iterrows():
                st.markdown(f"- {issue_row['issue_tag_split']} ({int(issue_row['article_count'])} articles)")

        st.markdown("**Interpretation**")
        st.write(comment)

        if constituents.get("available") and not constituents.get("rows", pd.DataFrame()).empty:
            with st.expander("Selected constituent return evidence"):
                display = constituents["rows"].copy()
                display["Return"] = display["Return"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                st.dataframe(display, use_container_width=True, hide_index=True)

        headlines = get_related_headlines(news_df, row["sector_key"], issues, max_headlines=3)
        if headlines:
            with st.expander("Small sample of related scraped headlines"):
                for headline in headlines:
                    st.markdown(f"- {headline}")

    col1, col2 = st.columns(2)
    with col1:
        render_issue_card("Best Performing Sector", best_sector_row, best_issues, best_comment, best_constituents, "green")
    with col2:
        render_issue_card("Weakest Sector", weakest_sector_row, weak_issues, weak_comment, weak_constituents, "red")

    st.info(
        "This brief uses rule-based issue tagging and selected constituent price evidence. "
        "It helps connect performance with news flow, but it should not be interpreted as proof of causality."
    )

def render_data_source_log():
    st.header("Data Source & Collection Log")
    st.caption(
        "This section makes the API and web-scraping workflow explicit for the project. "
        "It summarizes each dataset, collection method, generated file, and dashboard usage."
    )

    source_df = pd.DataFrame(
        [
            {
                "Data Item": "U.S. 10Y Treasury Yield",
                "Source / Method": "FRED DGS10 API",
                "Generated File": "timeseries_macro.csv",
                "Dashboard Usage": "REIT vs Equity vs Rates; Market Yield Spread; Sector Yield Spread",
            },
            {
                "Data Item": "Key REIT Constituent Prices",
                "Source / Method": "yfinance API",
                "Generated File": "Fetched dynamically in app.py",
                "Dashboard Usage": "Key Constituents Price Performance",
            },
            {
                "Data Item": "Sector News Headlines",
                "Source / Method": "Google News RSS scraping",
                "Generated File": "sector_news.csv",
                "Dashboard Usage": "Sector News Monitor; Issue Tag; Latest Articles",
            },
            {
                "Data Item": "FTSE EPRA Nareit Developed Factsheet",
                "Source / Method": "FTSE Russell PDF download + PDF parsing",
                "Generated File": "overview_latest.csv; country_weight_latest.csv",
                "Dashboard Usage": "Overview; Number of constituents; Net market cap; Country weight",
            },
            {
                "Data Item": "REIT Index / Sector / Region Index Data",
                "Source / Method": "Bloomberg Excel export",
                "Generated File": "timeseries_macro.csv; sector_returns.csv; region_returns.csv",
                "Dashboard Usage": "Market, regional, and sector performance",
            },
            {
                "Data Item": "Dividend Yield Data",
                "Source / Method": "Bloomberg Excel export",
                "Generated File": "sector_yields.csv",
                "Dashboard Usage": "REIT Dividend Yield Valuation",
            },
        ]
    )

    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.info(
        "The dashboard combines institutional market data from Bloomberg with public API and web-scraped data. "
        "FRED, yfinance, Google News RSS, and FTSE Russell factsheet parsing are used to strengthen the API/web-scraping component."
    )


# --------------------------------------------------
# Section: Data Refresh Guide
# --------------------------------------------------
def render_data_refresh_guide():
    st.header("Data Refresh Guide")
    st.caption("Run the following scripts from the project root folder when refreshing the dashboard data.")

    refresh_df = pd.DataFrame(
        [
            {
                "Step": 1,
                "Command": "python factsheet_download.py",
                "Purpose": "Download and parse the latest FTSE Russell factsheet PDF",
                "Outputs": "data/parsed/overview_latest.csv; data/parsed/country_weight_latest.csv",
            },
            {
                "Step": 2,
                "Command": "python data_download.py",
                "Purpose": "Process Bloomberg Excel data and collect FRED DGS10 API data",
                "Outputs": "timeseries_macro.csv; region_returns.csv; sector_returns.csv; sector_yields.csv; sector_ticker_universe.csv",
            },
            {
                "Step": 3,
                "Command": "python news_download.py",
                "Purpose": "Scrape recent Google News RSS articles using sector keywords and index member tickers",
                "Outputs": "data/sector_news.csv",
            },
            {
                "Step": 4,
                "Command": "streamlit run app.py",
                "Purpose": "Launch the dashboard locally",
                "Outputs": "Interactive Streamlit dashboard",
            },
        ]
    )

    st.dataframe(refresh_df, use_container_width=True, hide_index=True)

    with st.expander("Copy refresh commands"):
        st.code(
            "python factsheet_download.py\n"
            "python data_download.py\n"
            "python news_download.py\n"
            "streamlit run app.py",
            language="bash",
        )


# --------------------------------------------------
# Render
# --------------------------------------------------
render_download_report_panel()
st.divider()

render_overview()
st.divider()

render_macro()
st.divider()

render_region_performance()
st.divider()

render_sector_performance()
render_key_constituents_price_performance()
st.divider()

render_reit_dividend_yield_valuation()
st.divider()

render_sector_news()
st.divider()

render_data_source_log()
st.divider()

render_data_refresh_guide()
