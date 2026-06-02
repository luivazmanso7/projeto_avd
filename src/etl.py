from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    PRICE_QUARTILE_LABELS,
    PROCESSED_BOOKS_PATH,
    RATING_LABELS,
    RATING_MAP,
    RAW_BOOKS_PATH,
)


POSITIVE_TITLE_WORDS = {
    "amazing",
    "beautiful",
    "brilliant",
    "courage",
    "extraordinary",
    "good",
    "great",
    "happy",
    "hope",
    "light",
    "love",
    "magic",
    "paradise",
    "star",
    "success",
    "sweet",
    "wonder",
}

NEGATIVE_TITLE_WORDS = {
    "blood",
    "broken",
    "crisis",
    "dark",
    "dead",
    "death",
    "evil",
    "fear",
    "lost",
    "murder",
    "ruin",
    "sad",
    "secrets",
    "war",
}


def parse_price(value: object) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else np.nan


def parse_integer(value: object) -> int:
    if pd.isna(value):
        return 0
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else 0


def normalize_text(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    return re.sub(r"\s+", " ", str(value)).strip() or fallback


def title_tone_score(title: str) -> int:
    words = set(re.findall(r"[A-Za-z]+", title.lower()))
    return len(words & POSITIVE_TITLE_WORDS) - len(words & NEGATIVE_TITLE_WORDS)


def add_price_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    if df["price_gbp"].nunique() < 4:
        df["price_quartile"] = "Sem quartil"
        return df

    df["price_quartile"] = pd.qcut(
        df["price_gbp"].rank(method="first"),
        q=4,
        labels=PRICE_QUARTILE_LABELS,
    ).astype(str)
    return df


def add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df["price_gbp"].quantile(0.25)
    q3 = df["price_gbp"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["price_q1"] = q1
    df["price_q3"] = q3
    df["price_iqr"] = iqr
    df["is_global_price_outlier"] = df["price_gbp"].lt(lower_bound) | df["price_gbp"].gt(upper_bound)

    grouped_price = df.groupby("category")["price_gbp"]
    category_q1 = grouped_price.transform(lambda values: values.quantile(0.25))
    category_q3 = grouped_price.transform(lambda values: values.quantile(0.75))
    category_iqr = category_q3 - category_q1
    category_size = grouped_price.transform("count")
    category_lower = category_q1 - 1.5 * category_iqr
    category_upper = category_q3 + 1.5 * category_iqr

    df["category_price_q1"] = category_q1
    df["category_price_q3"] = category_q3
    df["category_price_iqr"] = category_iqr
    df["is_category_price_outlier"] = (
        category_size.ge(4)
        & category_iqr.gt(0)
        & (df["price_gbp"].lt(category_lower) | df["price_gbp"].gt(category_upper))
    )
    df["is_price_outlier"] = df["is_global_price_outlier"] | df["is_category_price_outlier"]
    return df


def transform_books(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [column.strip().lower() for column in df.columns]

    expected_columns = {
        "title": "",
        "catalogue_price": np.nan,
        "rating_text": np.nan,
        "catalogue_availability": "",
        "detail_url": "",
        "image_url": "",
        "category": "Sem categoria",
        "description": "",
        "upc": "",
        "product_type": "Books",
        "price_excl_tax": np.nan,
        "price_incl_tax": np.nan,
        "tax": np.nan,
        "availability_detail": "",
        "num_reviews": 0,
    }
    for column, default in expected_columns.items():
        if column not in df:
            df[column] = default

    string_columns = [
        "title",
        "catalogue_availability",
        "detail_url",
        "image_url",
        "category",
        "description",
        "upc",
        "product_type",
        "availability_detail",
    ]
    for column in string_columns:
        fallback = "Sem categoria" if column == "category" else ""
        df[column] = df[column].map(lambda value: normalize_text(value, fallback=fallback))

    df["price_gbp"] = df["price_incl_tax"].map(parse_price)
    df["catalogue_price_gbp"] = df["catalogue_price"].map(parse_price)
    df["price_excl_tax_gbp"] = df["price_excl_tax"].map(parse_price)
    df["tax_gbp"] = df["tax"].map(parse_price)
    df["price_gbp"] = df["price_gbp"].fillna(df["catalogue_price_gbp"])
    df["price_excl_tax_gbp"] = df["price_excl_tax_gbp"].fillna(df["price_gbp"])
    df["tax_gbp"] = df["tax_gbp"].fillna(0)

    df["rating"] = df["rating_text"].map(RATING_MAP)
    df["rating"] = df["rating"].fillna(df["rating"].median()).astype(int)
    df["rating_label"] = df["rating"].map(RATING_LABELS)

    df["stock_quantity"] = df["availability_detail"].map(parse_integer)
    fallback_stock = df["catalogue_availability"].str.contains("In stock", case=False, na=False)
    df.loc[df["stock_quantity"].eq(0) & fallback_stock, "stock_quantity"] = 1
    df["availability_status"] = np.where(df["stock_quantity"].gt(0), "Disponivel", "Sem estoque")
    df["num_reviews"] = df["num_reviews"].map(parse_integer)

    df = df.dropna(subset=["title", "price_gbp"]).drop_duplicates(subset=["upc", "detail_url"])

    df["title_word_count"] = df["title"].str.split().str.len().fillna(0).astype(int)
    df["title_char_count"] = df["title"].str.len()
    df["description_word_count"] = df["description"].str.split().str.len().fillna(0).astype(int)
    df["title_tone_score"] = df["title"].map(title_tone_score)
    df["stock_value_gbp"] = df["price_gbp"] * df["stock_quantity"]
    df["value_score"] = (df["rating"] / df["price_gbp"]) * 100
    df["tax_rate_pct"] = np.where(
        df["price_excl_tax_gbp"].gt(0),
        (df["tax_gbp"] / df["price_excl_tax_gbp"]) * 100,
        0,
    )

    category_median = df.groupby("category")["price_gbp"].transform("median")
    df["price_vs_category_median_pct"] = ((df["price_gbp"] / category_median) - 1) * 100

    df = add_price_quartiles(df)
    df = add_outlier_flags(df)
    df = df.sort_values(["category", "price_gbp", "title"]).reset_index(drop=True)
    return df


def save_processed_books(df: pd.DataFrame, output_path: Path = PROCESSED_BOOKS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpa e enriquece os dados coletados.")
    parser.add_argument("--input", type=Path, default=RAW_BOOKS_PATH, help="CSV bruto.")
    parser.add_argument("--output", type=Path, default=PROCESSED_BOOKS_PATH, help="CSV limpo.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    raw_df = pd.read_csv(args.input)
    clean_df = transform_books(raw_df)
    output_path = save_processed_books(clean_df, args.output)
    print(f"ETL concluido: {len(clean_df)} livros salvos em {output_path}")


if __name__ == "__main__":
    main()
