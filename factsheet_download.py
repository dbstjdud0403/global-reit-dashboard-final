import os
import re
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader


# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path("data")
PARSED_DIR = DATA_DIR / "parsed"
RAW_DIR = DATA_DIR / "raw"

DATA_DIR.mkdir(exist_ok=True)
PARSED_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

PDF_PATH = RAW_DIR / "ftse_epra_nareit_developed_factsheet.pdf"
TEXT_PATH = RAW_DIR / "factsheet_extracted_text.txt"
OVERVIEW_OUTPUT = PARSED_DIR / "overview_latest.csv"
COUNTRY_OUTPUT = PARSED_DIR / "country_weight_latest.csv"

# FTSE EPRA Nareit Developed Index factsheet
FACTSHEET_URL = (
    "https://research.ftserussell.com/Analytics/FactSheets/Home/"
    "DownloadSingleIssue?isManual=False&issueName=ENGL"
)


# --------------------------------------------------
# Generic helpers
# --------------------------------------------------
def clean_number(value):
    if value is None:
        return None
    value = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(value)
    except Exception:
        return None


def download_factsheet():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(FACTSHEET_URL, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise ValueError(
            "Downloaded file does not look like a PDF. "
            f"Content-Type: {content_type}"
        )

    PDF_PATH.write_bytes(response.content)
    print(f"Saved PDF: {PDF_PATH}")


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"Saved extracted text: {TEXT_PATH}")
    return text


# --------------------------------------------------
# Overview parser
# --------------------------------------------------
def parse_overview(full_text):
    rows = []

    # Number of Constituents can appear in different locations/labels.
    constituents_patterns = [
        r"Number\s+of\s+Constituents\s*[:\-]?\s*([\d,]+)",
        r"No\.?\s+of\s+Constituents\s*[:\-]?\s*([\d,]+)",
        r"Constituents\s*[:\-]?\s*([\d,]+)",
    ]

    constituents_value = None
    for pattern in constituents_patterns:
        m = re.search(pattern, full_text, flags=re.IGNORECASE)
        if m:
            constituents_value = clean_number(m.group(1))
            break

    if constituents_value is not None:
        rows.append({"metric": "Number of Constituents", "value": int(constituents_value), "unit": ""})
    else:
        print("Warning: Number of Constituents not found.")

    # Net Market Cap / Net MCap can be in USDm/USDbn.
    market_cap_patterns = [
        r"Net\s+Market\s+Cap(?:italisation|italization)?\s*\(?\s*(USD\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
        r"Net\s+MCap\s*\(?\s*(USD\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
        r"Market\s+Cap(?:italisation|italization)?\s*\(?\s*(USD\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
    ]

    market_cap_value = None
    unit_label = "USD tn"
    for pattern in market_cap_patterns:
        m = re.search(pattern, full_text, flags=re.IGNORECASE)
        if m:
            unit_raw = (m.group(1) or "").lower().replace(" ", "")
            value = clean_number(m.group(2))
            if value is None:
                continue
            if "million" in unit_raw or unit_raw.endswith("m"):
                market_cap_value = value / 1_000_000
            elif "billion" in unit_raw or "bn" in unit_raw or unit_raw.endswith("b"):
                market_cap_value = value / 1_000
            elif "trillion" in unit_raw or "tn" in unit_raw or unit_raw.endswith("t"):
                market_cap_value = value
            else:
                # FTSE factsheets often show USDm in tables. Use magnitude heuristic.
                if value > 100_000:
                    market_cap_value = value / 1_000_000
                elif value > 100:
                    market_cap_value = value / 1_000
                else:
                    market_cap_value = value
            break

    if market_cap_value is not None:
        rows.append({"metric": "Net Market Cap", "value": round(market_cap_value, 2), "unit": unit_label})
    else:
        print("Warning: Net Market Cap not found.")

    return pd.DataFrame(rows)


# --------------------------------------------------
# Country weight parser
# --------------------------------------------------
def parse_country_weights(full_text):
    """
    Parse FTSE factsheet country breakdown.

    Important fix: FTSE often uses "USA" rather than "United States".
    Previous parsers that only searched for "United States" missed the largest country,
    causing Japan to appear as the largest slice after Plotly re-normalisation.
    """
    country_aliases = {
        "USA": "United States",
        "United States": "United States",
        "Japan": "Japan",
        "Australia": "Australia",
        "United Kingdom": "United Kingdom",
        "UK": "United Kingdom",
        "Canada": "Canada",
        "Singapore": "Singapore",
        "France": "France",
        "Hong Kong": "Hong Kong",
        "Germany": "Germany",
        "Switzerland": "Switzerland",
        "Sweden": "Sweden",
        "Spain": "Spain",
        "Belgium": "Belgium",
        "Netherlands": "Netherlands",
        "Italy": "Italy",
        "New Zealand": "New Zealand",
        "Finland": "Finland",
        "Ireland": "Ireland",
        "Austria": "Austria",
        "Norway": "Norway",
        "Israel": "Israel",
        "Denmark": "Denmark",
    }

    # Prefer the specific country breakdown block if it can be found.
    section_match = re.search(
        r"Country\s+Breakdown(.*?)(?:Totals\s+[\d,]+\s+[\d,]+(?:\.\d+)?\s+100\.?0*|Index\s+Characteristics|Top\s+10|Performance|Disclaimer|Source)",
        full_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    section_text = section_match.group(1) if section_match else full_text

    rows = []

    for raw_country, clean_country in country_aliases.items():
        # Common FTSE row format after text extraction:
        # USA 123 696,406 54.20
        # Japan 59 158,xxx 12.31
        # Capture the final numeric value after number-of-constituents and net mcap.
        patterns = [
            rf"\b{re.escape(raw_country)}\b\s+\d+\s+[\d,]+(?:\.\d+)?\s+([\d]+(?:\.\d+)?)",
            rf"\b{re.escape(raw_country)}\b\s+([\d]+(?:\.\d+)?)\s*%",
            rf"\b{re.escape(raw_country)}\b\s+([\d]+(?:\.\d+)?)\b",
        ]

        matched_weight = None
        for pattern in patterns:
            m = re.search(pattern, section_text, flags=re.IGNORECASE)
            if m:
                matched_weight = clean_number(m.group(1))
                # Skip obviously non-weight values from the loose fallback.
                if matched_weight is not None and 0 <= matched_weight <= 100:
                    break
                matched_weight = None

        if matched_weight is not None:
            rows.append({"country": clean_country, "weight": matched_weight})

    country_df = pd.DataFrame(rows)

    if not country_df.empty:
        country_df = (
            country_df.groupby("country", as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
        )

    total_weight = country_df["weight"].sum() if not country_df.empty else 0
    has_us = (country_df["country"].eq("United States").any() if not country_df.empty else False)

    if not has_us:
        print("Warning: United States/USA was not parsed. Check factsheet_extracted_text.txt.")
    if total_weight < 95:
        print(f"Warning: Country weights sum to only {total_weight:.2f}. Parser may have missed some countries.")

    return country_df


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Downloading FTSE EPRA Nareit Developed factsheet...")
    download_factsheet()

    print("Extracting PDF text...")
    full_text = extract_pdf_text(PDF_PATH)

    print("Parsing overview data...")
    overview_df = parse_overview(full_text)
    if not overview_df.empty:
        overview_df.to_csv(OVERVIEW_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Saved: {OVERVIEW_OUTPUT}")
        print(overview_df)
    else:
        print("Warning: overview_latest.csv was not created.")

    print("Parsing country weights...")
    country_df = parse_country_weights(full_text)
    if not country_df.empty:
        country_df.to_csv(COUNTRY_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Saved: {COUNTRY_OUTPUT}")
        print(country_df)
        print(f"Country weight sum: {country_df['weight'].sum():.2f}")
    else:
        print("Warning: country_weight_latest.csv was not created.")
        print(f"Check extracted text: {TEXT_PATH}")

    print("\nFactsheet download / parsing completed.")


if __name__ == "__main__":
    main()
