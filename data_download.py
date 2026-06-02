import os
import urllib.parse
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
os.makedirs("data", exist_ok=True)
DATA_DIR = Path("data")
BBG_FILE = DATA_DIR / "REITs Data.xlsx"

if not BBG_FILE.exists():
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {BBG_FILE}")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    """Handle normal datetimes and Excel serial numbers."""
    dt = pd.to_datetime(s, errors="coerce")
    # If many dates are missing and input is numeric, try Excel serial conversion.
    if dt.notna().sum() < max(3, len(s) * 0.3):
        numeric = pd.to_numeric(s, errors="coerce")
        excel_dt = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        dt = dt.fillna(excel_dt)
    return dt


def read_bloomberg_value_sheet(file_path, sheet_name):
    """Read Bloomberg *_Value time-series sheets into date + ticker columns."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    tickers = raw.iloc[3, 1:].dropna().astype(str).str.strip().tolist()
    data = raw.iloc[6:, : len(tickers) + 1].copy()
    data.columns = ["date"] + tickers
    data["date"] = parse_date_series(data["date"])
    data = data.dropna(subset=["date"]).sort_values("date")
    for col in data.columns:
        if col != "date":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def read_dividend_yield_sheet(file_path, sheet_name="Dividend 12Month Yield_Value"):
    """Read Dividend 12Month Yield_Value using sector names in row 3."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    sector_names = raw.iloc[2, 1:].dropna().astype(str).str.strip().tolist()
    data = raw.iloc[6:, : len(sector_names) + 1].copy()
    data.columns = ["date"] + sector_names
    data["date"] = parse_date_series(data["date"])
    data = data.dropna(subset=["date"]).sort_values("date")
    for col in data.columns:
        if col != "date":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def fetch_fred_series(series_id, start_date="2015-12-31"):
    """FRED CSV download. DGS10 = U.S. 10-Year Treasury Constant Maturity Rate."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + f"id={urllib.parse.quote(series_id)}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id.lower()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= pd.to_datetime(start_date)]
    df = df.sort_values("date")
    return df


def calc_return_by_date(series, start_date, end_date):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    start_values = s[s.index <= start_date]
    end_values = s[s.index <= end_date]
    if start_values.empty or end_values.empty:
        return None
    start = start_values.iloc[-1]
    end = end_values.iloc[-1]
    if pd.isna(start) or start == 0:
        return None
    return (end / start - 1) * 100


def bloomberg_to_search_ticker(bloomberg_ticker):
    """Basic ticker for Google News/yfinance search. Full conversion happens in app.py."""
    if pd.isna(bloomberg_ticker):
        return None
    t = str(bloomberg_ticker).strip()
    if t == "" or t.lower().startswith("#n/a"):
        return None
    return t.split()[0]

# --------------------------------------------------
# Index maps
# --------------------------------------------------
SECTOR_INDEX_MAP = {
    "TENGVU Index": "Diversified",
    "TENGEU Index": "Healthcare",
    "TENGIU Index": "Industrial",
    "TENGMU Index": "Industrial / Office",
    "TENGOU Index": "Lodging / Resorts",
    "TENGFU Index": "Office",
    "TENGAU Index": "Residential",
    "TENGTU Index": "Retail",
    "TENGSU Index": "Self Storage",
    "ENGCT Index": "Data Center",
    "ENGYT Index": "Specialty",
}

REGION_INDEX_MAP = {
    "TERNAU Index": "North America",
    "UNUS Index": "United States",
    "RXUK Index": "United Kingdom",
    "ELUK Index": "Europe ex UK",
    "TEGAXU Index": "Asia Pacific ex Japan",
    "ELJP Index": "Japan",
    "ELHK Index": "Hong Kong",
}

# --------------------------------------------------
# 1) Macro / index data
# --------------------------------------------------
index_df = read_bloomberg_value_sheet(BBG_FILE, "Index Performance_value")
try:
    ust_df = fetch_fred_series("DGS10", start_date="2015-12-31")
    ust_source = "FRED DGS10 API"
except Exception as e:
    print(f"Warning: FRED download failed ({e}). Falling back to US 10Y Treasury Yield_Value sheet.")
    ust_raw = read_bloomberg_value_sheet(BBG_FILE, "US 10Y Treasury Yield_Value")
    ust_cols = [c for c in ust_raw.columns if c != "date"]
    if not ust_cols:
        raise KeyError("US 10Y Treasury Yield_Value 시트에서 10Y 컬럼을 찾지 못했습니다.")
    ust_df = ust_raw[["date", ust_cols[0]]].rename(columns={ust_cols[0]: "dgs10"})
    ust_source = "Bloomberg fallback"

dividend_df = read_dividend_yield_sheet(BBG_FILE, "Dividend 12Month Yield_Value")

ust_col = [c for c in ust_df.columns if c != "date"][0]

required_macro_cols = ["TRNGLU Index", "SPTR500N Index", "M1WO Index"]
missing_macro_cols = [c for c in required_macro_cols if c not in index_df.columns]
if missing_macro_cols:
    raise KeyError(f"Index Performance_value에서 필요한 컬럼이 없습니다: {missing_macro_cols}")

macro = index_df[["date"] + required_macro_cols].copy()
macro = macro.rename(
    columns={
        "TRNGLU Index": "FTSE EPRA Nareit Developed",
        "SPTR500N Index": "S&P 500",
        "M1WO Index": "MSCI World",
    }
)
macro = macro.merge(ust_df[["date", ust_col]].rename(columns={ust_col: "UST 10Y"}), on="date", how="left")

if "EPRA/NAREIT Developed" in dividend_df.columns:
    macro = macro.merge(
        dividend_df[["date", "EPRA/NAREIT Developed"]].rename(columns={"EPRA/NAREIT Developed": "Dividend Yield"}),
        on="date",
        how="left",
    )
else:
    raise KeyError("Dividend 12Month Yield_Value에서 'EPRA/NAREIT Developed' 컬럼을 찾지 못했습니다.")

macro = macro.dropna(subset=["FTSE EPRA Nareit Developed"]).sort_values("date")
macro.to_csv(DATA_DIR / "timeseries_macro.csv", index=False, encoding="utf-8-sig")
print("Saved: data/timeseries_macro.csv")

# --------------------------------------------------
# 2) Sector returns
# --------------------------------------------------
sector_cols = [ticker for ticker in SECTOR_INDEX_MAP if ticker in index_df.columns]
sector_source = index_df[["date"] + sector_cols].set_index("date").sort_index()
sector_end_date = sector_source.index.max()
sector_rows = []
for ticker in sector_cols:
    series = sector_source[ticker]
    sector_rows.append(
        {
            "sector": SECTOR_INDEX_MAP[ticker],
            "ticker": ticker,
            "as_of_date": sector_end_date.strftime("%Y-%m-%d"),
            "return_1m": calc_return_by_date(series, sector_end_date - pd.DateOffset(months=1), sector_end_date),
            "return_ytd": calc_return_by_date(series, pd.Timestamp(year=sector_end_date.year, month=1, day=1), sector_end_date),
            "return_12m": calc_return_by_date(series, sector_end_date - pd.DateOffset(months=12), sector_end_date),
            "source": "Bloomberg sector index",
        }
    )
sector_returns = pd.DataFrame(sector_rows).dropna(subset=["return_1m", "return_ytd", "return_12m"], how="all")
sector_returns.to_csv(DATA_DIR / "sector_returns.csv", index=False, encoding="utf-8-sig")
print("Saved: data/sector_returns.csv")

# --------------------------------------------------
# 3) Region returns
# --------------------------------------------------
region_cols = [ticker for ticker in REGION_INDEX_MAP if ticker in index_df.columns]
region_source = index_df[["date"] + region_cols].set_index("date").sort_index()
region_end_date = region_source.index.max()
region_rows = []
for ticker in region_cols:
    series = region_source[ticker]
    region_rows.append(
        {
            "region": REGION_INDEX_MAP[ticker],
            "ticker": ticker,
            "as_of_date": region_end_date.strftime("%Y-%m-%d"),
            "return_1m": calc_return_by_date(series, region_end_date - pd.DateOffset(months=1), region_end_date),
            "return_ytd": calc_return_by_date(series, pd.Timestamp(year=region_end_date.year, month=1, day=1), region_end_date),
            "return_12m": calc_return_by_date(series, region_end_date - pd.DateOffset(months=12), region_end_date),
            "source": "Bloomberg regional index",
        }
    )
region_returns = pd.DataFrame(region_rows).dropna(subset=["return_1m", "return_ytd", "return_12m"], how="all")
region_returns.to_csv(DATA_DIR / "region_returns.csv", index=False, encoding="utf-8-sig")
print("Saved: data/region_returns.csv")

# --------------------------------------------------
# 4) Sector dividend yield data
# --------------------------------------------------
sector_yield_cols = [col for col in dividend_df.columns if col not in ["date", "EPRA/NAREIT Developed"]]
sector_yields = dividend_df[["date"] + sector_yield_cols].dropna(subset=sector_yield_cols, how="all").sort_values("date")
sector_yields.to_csv(DATA_DIR / "sector_yields.csv", index=False, encoding="utf-8-sig")
print("Saved: data/sector_yields.csv")

# --------------------------------------------------
# 5) Sector ticker universe from Index Members_Value
# --------------------------------------------------
def build_sector_ticker_universe():
    raw = pd.read_excel(BBG_FILE, sheet_name="Index Members_Value", header=None)
    rows = []

    # New structure: every 4 columns are [Sector Index, Member Ticker, INDX_MWEIGHT, blank]
    for col in range(0, raw.shape[1], 4):
        index_ticker = raw.iloc[2, col] if col < raw.shape[1] else None
        if pd.isna(index_ticker):
            continue
        index_ticker = str(index_ticker).strip()
        if index_ticker not in SECTOR_INDEX_MAP:
            continue

        sector_name = SECTOR_INDEX_MAP[index_ticker]
        member_col = col + 1
        weight_col = col + 2
        if member_col >= raw.shape[1]:
            continue

        block = raw.iloc[2:, [member_col, weight_col] if weight_col < raw.shape[1] else [member_col]].copy()
        if weight_col < raw.shape[1]:
            block.columns = ["bloomberg_ticker", "index_weight"]
        else:
            block.columns = ["bloomberg_ticker"]
            block["index_weight"] = pd.NA

        block["bloomberg_ticker"] = block["bloomberg_ticker"].astype(str).str.strip()
        block = block[~block["bloomberg_ticker"].str.lower().isin(["", "nan", "none"])]
        block = block[~block["bloomberg_ticker"].str.lower().str.startswith("#n/a")]
        block["index_weight"] = pd.to_numeric(block["index_weight"], errors="coerce")
        block["source_order"] = range(1, len(block) + 1)
        block["sector"] = sector_name
        block["sector_index"] = index_ticker
        block["search_ticker"] = block["bloomberg_ticker"].apply(bloomberg_to_search_ticker)

        # Rank by index weight if available, otherwise original order.
        if block["index_weight"].notna().any():
            block = block.sort_values(["index_weight", "source_order"], ascending=[False, True], na_position="last")
        else:
            block = block.sort_values("source_order")
        block["rank"] = range(1, len(block) + 1)
        rows.append(block[["sector", "sector_index", "rank", "source_order", "bloomberg_ticker", "search_ticker", "index_weight"]])

    if not rows:
        return pd.DataFrame(columns=["sector", "sector_index", "rank", "source_order", "bloomberg_ticker", "search_ticker", "index_weight"])

    universe = pd.concat(rows, ignore_index=True)
    return universe

sector_ticker_universe = build_sector_ticker_universe()
sector_ticker_universe.to_csv(DATA_DIR / "sector_ticker_universe.csv", index=False, encoding="utf-8-sig")
print("Saved: data/sector_ticker_universe.csv")
print(sector_ticker_universe.head(20))

print("\nData download / preprocessing completed.")
