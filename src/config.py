from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_BOOKS_PATH = RAW_DIR / "books_raw.csv"
PROCESSED_BOOKS_PATH = PROCESSED_DIR / "books_clean.csv"
ANALYSIS_REPORT_PATH = PROCESSED_DIR / "analysis_report.json"

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/page-{page}.html"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

RATING_LABELS = {
    1: "1 estrela",
    2: "2 estrelas",
    3: "3 estrelas",
    4: "4 estrelas",
    5: "5 estrelas",
}

PRICE_QUARTILE_LABELS = [
    "Q1 - Mais barato",
    "Q2 - Abaixo da mediana",
    "Q3 - Acima da mediana",
    "Q4 - Mais caro",
]
