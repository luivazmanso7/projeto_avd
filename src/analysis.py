from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import ANALYSIS_REPORT_PATH, PROCESSED_BOOKS_PATH


NUMERIC_FEATURES = [
    "rating",
    "stock_quantity",
    "title_word_count",
    "title_char_count",
    "description_word_count",
    "title_tone_score",
    "num_reviews",
]

CATEGORICAL_FEATURES = ["category", "availability_status"]


def pearson_correlations(
    df: pd.DataFrame,
    target: str = "price_gbp",
    columns: Iterable[str] = NUMERIC_FEATURES,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for column in columns:
        pair = df[[target, column]].dropna()
        if len(pair) < 3 or pair[target].nunique() < 2 or pair[column].nunique() < 2:
            continue

        coefficient, p_value = stats.pearsonr(pair[target], pair[column])
        rows.append(
            {
                "variavel": column,
                "correlacao_pearson": coefficient,
                "p_valor": p_value,
                "n": len(pair),
                "forca": interpret_correlation(coefficient),
            }
        )
    result = pd.DataFrame(
        rows,
        columns=["variavel", "correlacao_pearson", "p_valor", "n", "forca"],
    )
    if result.empty:
        return result
    return result.sort_values("correlacao_pearson", key=lambda s: s.abs(), ascending=False)


def interpret_correlation(value: float) -> str:
    absolute = abs(value)
    if absolute >= 0.70:
        return "forte"
    if absolute >= 0.40:
        return "moderada"
    if absolute >= 0.20:
        return "fraca"
    return "muito fraca"


def anova_by_group(
    df: pd.DataFrame,
    group_col: str = "rating_label",
    value_col: str = "price_gbp",
) -> dict[str, float | int | str]:
    groups = [
        group[value_col].dropna().to_numpy()
        for _, group in df.groupby(group_col)
        if group[value_col].dropna().shape[0] >= 2
    ]
    if len(groups) < 2:
        return {
            "group_col": group_col,
            "value_col": value_col,
            "f_statistic": np.nan,
            "p_value": np.nan,
            "groups": len(groups),
            "interpretation": "Amostra insuficiente",
        }

    f_statistic, p_value = stats.f_oneway(*groups)
    return {
        "group_col": group_col,
        "value_col": value_col,
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "groups": len(groups),
        "interpretation": (
            "Diferencas significativas entre grupos"
            if p_value < 0.05
            else "Sem evidencia estatistica de diferenca entre grupos"
        ),
    }


def quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("price_quartile", observed=False)
        .agg(
            livros=("title", "count"),
            preco_minimo=("price_gbp", "min"),
            preco_mediano=("price_gbp", "median"),
            preco_maximo=("price_gbp", "max"),
            avaliacao_media=("rating", "mean"),
            estoque_mediano=("stock_quantity", "median"),
        )
        .reset_index()
    )


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")
        .agg(
            livros=("title", "count"),
            preco_medio=("price_gbp", "mean"),
            preco_mediano=("price_gbp", "median"),
            preco_minimo=("price_gbp", "min"),
            preco_maximo=("price_gbp", "max"),
            avaliacao_media=("rating", "mean"),
            estoque_total=("stock_quantity", "sum"),
            valor_estoque=("stock_value_gbp", "sum"),
            outliers=("is_price_outlier", "sum"),
        )
        .reset_index()
        .sort_values(["livros", "preco_mediano"], ascending=[False, False])
    )


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "title",
        "category",
        "price_gbp",
        "rating",
        "stock_quantity",
        "price_vs_category_median_pct",
    ]
    return df.loc[df["is_price_outlier"], columns].sort_values("price_gbp", ascending=False)


def build_ml_model(df: pd.DataFrame) -> dict[str, object]:
    model_df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["price_gbp"]].dropna()
    if len(model_df) < 40:
        return {"available": False, "reason": "Amostra insuficiente para treino e teste."}

    x = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df["price_gbp"]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=240,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    fitted_preprocessor = model.named_steps["preprocess"]
    categorical_pipeline = fitted_preprocessor.named_transformers_["categorical"]
    categorical_names = categorical_pipeline.named_steps["encoder"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    feature_names = np.concatenate([np.array(NUMERIC_FEATURES), categorical_names])
    importances = model.named_steps["regressor"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    return {
        "available": True,
        "algorithm": "RandomForestRegressor",
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "feature_importance": importance_df,
    }


def build_analysis_report(df: pd.DataFrame) -> dict[str, object]:
    return {
        "kpis": {
            "livros": int(len(df)),
            "categorias": int(df["category"].nunique()),
            "preco_medio": float(df["price_gbp"].mean()),
            "preco_mediano": float(df["price_gbp"].median()),
            "avaliacao_media": float(df["rating"].mean()),
            "outliers": int(df["is_price_outlier"].sum()),
        },
        "pearson": pearson_correlations(df),
        "anova_rating": anova_by_group(df, group_col="rating_label"),
        "anova_category": anova_by_group(df, group_col="category"),
        "quartiles": quartile_summary(df),
        "categories": category_summary(df),
        "outliers": outlier_table(df),
        "ml": build_ml_model(df),
    }


def to_jsonable(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_analysis_report(report: dict[str, object], output_path: Path = ANALYSIS_REPORT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera relatorio estatistico e modelo ML.")
    parser.add_argument("--input", type=Path, default=PROCESSED_BOOKS_PATH, help="CSV limpo.")
    parser.add_argument("--output", type=Path, default=ANALYSIS_REPORT_PATH, help="JSON de saida.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    report = build_analysis_report(df)
    output_path = save_analysis_report(report, args.output)
    print(f"Analise concluida: relatorio salvo em {output_path}")


if __name__ == "__main__":
    main()
