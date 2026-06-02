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

OVERVIEW_OUTPUT = PARSED_DIR / "overview_latest.csv"
COUNTRY_OUTPUT = PARSED_DIR / "country_weight_latest.csv"


# --------------------------------------------------
# FTSE Russell factsheet URL
# issueName=ENGL: FTSE EPRA Nareit Developed Index
# --------------------------------------------------
FACTSHEET_URL = (
    "https://research.ftserussell.com/Analytics/FactSheets/Home/"
    "DownloadSingleIssue?isManual=False&issueName=ENGL"
)


# --------------------------------------------------
# Download PDF
# --------------------------------------------------
def download_factsheet():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        FACTSHEET_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise ValueError(
            "다운로드한 파일이 PDF가 아닌 것 같습니다. "
            f"Content-Type: {content_type}"
        )

    PDF_PATH.write_bytes(response.content)

    print(f"Saved PDF: {PDF_PATH}")


# --------------------------------------------------
# Extract text
# --------------------------------------------------
def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))

    pages_text = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)

    # 디버깅용 텍스트 저장
    debug_path = RAW_DIR / "factsheet_extracted_text.txt"
    debug_path.write_text(full_text, encoding="utf-8")

    print(f"Saved extracted text: {debug_path}")

    return full_text


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_number(value):
    if value is None:
        return None

    value = str(value).replace(",", "").strip()

    try:
        return float(value)
    except Exception:
        return None


def find_first_match(patterns, text, flags=re.IGNORECASE | re.DOTALL):
    for pattern in patterns:
        match = re.search(pattern, text, flags)

        if match:
            return match

    return None


# --------------------------------------------------
# Parse overview
# --------------------------------------------------
def parse_overview(full_text):
    """
    overview_latest.csv 생성용:
    - Number of Constituents
    - Net Market Cap

    PDF 레이아웃이 바뀔 수 있어서 여러 regex 패턴을 순차적으로 시도.
    """

    # Number of Constituents
    constituents_patterns = [
        r"number\s+of\s+constituents\s*[:\-]?\s*([\d,]+)",
        r"no\.?\s+of\s+constituents\s*[:\-]?\s*([\d,]+)",
        r"constituents\s*[:\-]?\s*([\d,]+)",
    ]

    constituents_match = find_first_match(constituents_patterns, full_text)
    constituents_value = None

    if constituents_match:
        constituents_value = clean_number(constituents_match.group(1))

    # Net Market Cap
    # 보통 factsheet에는 USDm, USD mn, USD million 형태로 표시될 수 있음.
    market_cap_patterns = [
        r"net\s+market\s+cap(?:italisation|italization)?\s*\(?\s*(usd\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
        r"net\s+mcap\s*\(?\s*(usd\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
        r"market\s+cap(?:italisation|italization)?\s*\(?\s*(usd\s*[mbtnmillion]*)?\s*\)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
    ]

    market_cap_match = find_first_match(market_cap_patterns, full_text)

    market_cap_value = None
    market_cap_unit = "USD tn"

    if market_cap_match:
        unit_raw = market_cap_match.group(1) or ""
        value_raw = market_cap_match.group(2)

        value = clean_number(value_raw)

        unit_lower = unit_raw.lower().replace(" ", "")

        if value is not None:
            # USDm이면 trillion으로 변환
            if "m" in unit_lower or "million" in unit_lower:
                market_cap_value = value / 1_000_000
                market_cap_unit = "USD tn"
            # USDbn이면 trillion으로 변환
            elif "b" in unit_lower or "bn" in unit_lower or "billion" in unit_lower:
                market_cap_value = value / 1_000
                market_cap_unit = "USD tn"
            # trillion이면 그대로
            elif "t" in unit_lower or "tn" in unit_lower or "trillion" in unit_lower:
                market_cap_value = value
                market_cap_unit = "USD tn"
            else:
                # 단위가 안 잡히면 값 크기로 추정
                if value > 100_000:
                    market_cap_value = value / 1_000_000
                    market_cap_unit = "USD tn"
                elif value > 100:
                    market_cap_value = value / 1_000
                    market_cap_unit = "USD tn"
                else:
                    market_cap_value = value
                    market_cap_unit = "USD tn"

    rows = []

    if constituents_value is not None:
        rows.append(
            {
                "metric": "Number of Constituents",
                "value": int(constituents_value),
                "unit": ""
            }
        )
    else:
        print("Warning: Number of Constituents를 찾지 못했습니다.")

    if market_cap_value is not None:
        rows.append(
            {
                "metric": "Net Market Cap",
                "value": round(market_cap_value, 2),
                "unit": market_cap_unit
            }
        )
    else:
        print("Warning: Net Market Cap을 찾지 못했습니다.")

    overview_df = pd.DataFrame(rows)

    return overview_df


# --------------------------------------------------
# Parse country weight
# --------------------------------------------------
def parse_country_weights(full_text):
    """
    country_weight_latest.csv 생성용.

    PDF text extraction은 표 구조가 깨질 수 있어서,
    대표 국가명 + 숫자 패턴을 이용해 country weight를 추출.
    """

    countries = [
        "United States",
        "Japan",
        "Australia",
        "United Kingdom",
        "Canada",
        "Singapore",
        "France",
        "Hong Kong",
        "Germany",
        "Switzerland",
        "Sweden",
        "Spain",
        "Belgium",
        "Netherlands",
        "Italy",
        "New Zealand",
        "Finland",
        "Ireland",
        "Austria",
        "Norway",
        "Other",
    ]

    rows = []

    # Country Weights 주변 텍스트만 우선 사용
    section_text = full_text

    section_match = re.search(
        r"(country\s+weights?.{0,3000})",
        full_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if section_match:
        section_text = section_match.group(1)

    for country in countries:
        # 예: United States 64.52
        pattern = rf"{re.escape(country)}\s+([\d]+(?:\.\d+)?)"

        match = re.search(
            pattern,
            section_text,
            flags=re.IGNORECASE
        )

        if match:
            weight = clean_number(match.group(1))

            if weight is not None:
                rows.append(
                    {
                        "country": country,
                        "weight": weight
                    }
                )

    country_df = pd.DataFrame(rows)

    # 만약 country section에서 못 찾으면 전체 text에서 한 번 더 시도
    if country_df.empty:
        rows = []

        for country in countries:
            pattern = rf"{re.escape(country)}\s+([\d]+(?:\.\d+)?)"

            match = re.search(
                pattern,
                full_text,
                flags=re.IGNORECASE
            )

            if match:
                weight = clean_number(match.group(1))

                if weight is not None:
                    rows.append(
                        {
                            "country": country,
                            "weight": weight
                        }
                    )

        country_df = pd.DataFrame(rows)

    if not country_df.empty:
        country_df = country_df.drop_duplicates(subset=["country"])
        country_df = country_df.sort_values("weight", ascending=False)

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
        print("Warning: overview_latest.csv를 생성하지 못했습니다.")

    print("Parsing country weights...")
    country_df = parse_country_weights(full_text)

    if not country_df.empty:
        country_df.to_csv(COUNTRY_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Saved: {COUNTRY_OUTPUT}")
        print(country_df)
    else:
        print("Warning: country_weight_latest.csv를 생성하지 못했습니다.")
        print("data/raw/factsheet_extracted_text.txt를 열어서 Country Weights 텍스트 구조를 확인해주세요.")

    print("\nFactsheet download / parsing completed.")


if __name__ == "__main__":
    main()