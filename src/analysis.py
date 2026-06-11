from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
TEXT_FEATURES = ["title", "description"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
MIN_MODEL_ROWS = 40

FEATURE_LABELS = {
    "rating": "Avaliação",
    "stock_quantity": "Estoque declarado",
    "title_word_count": "Palavras no título",
    "title_char_count": "Caracteres no título",
    "description_word_count": "Palavras na descrição",
    "title_tone_score": "Tom do título",
    "num_reviews": "Reviews",
    "category": "Categoria",
    "availability_status": "Disponibilidade",
    "title": "Texto do título",
    "description": "Texto da descrição",
}


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


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "title_text",
                TfidfVectorizer(
                    max_features=120,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                "title",
            ),
            (
                "description_text",
                TfidfVectorizer(
                    max_features=280,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                "description",
            ),
        ],
        sparse_threshold=0.0,
    )


def candidate_regressors() -> dict[str, object]:
    return {
        "Ridge regularizado": Ridge(alpha=8.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=320,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.045,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
    }


def build_regression_pipeline(regressor: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def build_ml_model(df: pd.DataFrame, include_estimator: bool = False) -> dict[str, object]:
    metadata_columns = ["title"]
    model_columns = MODEL_FEATURES + ["price_gbp"]
    model_df = df[model_columns].dropna()
    if len(model_df) < MIN_MODEL_ROWS:
        return {"available": False, "reason": "Amostra insuficiente para treino e teste."}

    x = model_df[MODEL_FEATURES]
    y = model_df["price_gbp"]
    metadata = model_df[metadata_columns]

    x_train, x_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        x,
        y,
        metadata,
        test_size=0.25,
        random_state=42,
    )

    cv_folds = min(5, max(2, len(x_train) // 20))
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    metrics: list[dict[str, float | int | str | bool]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, regressor in candidate_regressors().items():
        pipeline = build_regression_pipeline(regressor)
        cv_result = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            error_score=np.nan,
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted_models[name] = pipeline

        metrics.append(
            {
                "modelo": name,
                "mae_cv": float(-np.nanmean(cv_result["test_mae"])),
                "rmse_cv": float(-np.nanmean(cv_result["test_rmse"])),
                "r2_cv": float(np.nanmean(cv_result["test_r2"])),
                "mae_teste": float(mean_absolute_error(y_test, predictions)),
                "rmse_teste": float(root_mean_squared_error(y_test, predictions)),
                "r2_teste": float(r2_score(y_test, predictions)),
                "folds_cv": int(cv_folds),
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values(["mae_cv", "mae_teste"]).reset_index(drop=True)
    best_metric_name = str(metrics_df.iloc[0]["modelo"])
    selected_name = best_metric_name
    metrics_df["selecionado"] = metrics_df["modelo"].eq(selected_name)

    selected_model = fitted_models[selected_name]
    selected_predictions = selected_model.predict(x_test)
    residuals = metadata_test.reset_index(drop=True).assign(
        category=x_test["category"].reset_index(drop=True),
        preco_real=y_test.reset_index(drop=True),
        preco_previsto=selected_predictions,
    )
    residuals["erro"] = residuals["preco_previsto"] - residuals["preco_real"]
    residuals["erro_absoluto"] = residuals["erro"].abs()
    residuals = residuals.sort_values("erro_absoluto", ascending=False).reset_index(drop=True)

    permutation = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": [FEATURE_LABELS.get(column, column) for column in x_test.columns],
                "importance": permutation.importances_mean.clip(min=0),
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    selected_mae = float(metrics_df.loc[metrics_df["selecionado"], "mae_teste"].iloc[0])

    result: dict[str, object] = {
        "available": True,
        "algorithm": selected_name,
        "best_metric_algorithm": best_metric_name,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cv_folds": int(cv_folds),
        "mae": selected_mae,
        "rmse": float(metrics_df.loc[metrics_df["selecionado"], "rmse_teste"].iloc[0]),
        "r2": float(metrics_df.loc[metrics_df["selecionado"], "r2_teste"].iloc[0]),
        "model_metrics": metrics_df,
        "feature_importance": importance_df,
        "residuals": residuals,
    }
    if include_estimator:
        final_model = build_regression_pipeline(candidate_regressors()[selected_name])
        final_model.fit(x, y)
        result["estimator"] = final_model
    return result


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
