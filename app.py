from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    anova_by_group,
    build_ml_model,
    category_summary,
    outlier_table,
    pearson_correlations,
    quartile_summary,
)
from src.config import PROCESSED_BOOKS_PATH, RAW_BOOKS_PATH
from src.etl import save_processed_books, title_tone_score, transform_books
from src.scraper import save_raw_books, scrape_books
from src.visuals import (
    category_median_bar,
    correlation_bar,
    feature_importance_bar,
    model_mae_bar,
    predicted_vs_actual_scatter,
    price_histogram,
    quartile_bar,
    rating_price_box,
    stock_price_scatter,
)


st.set_page_config(
    page_title="Radar Editorial AVD",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: #F6F8FA;
        color: #1F2933;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #667085;
    }
    [data-testid="stMetricValue"] {
        color: #1F2933;
        font-size: 1.7rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
</style>
"""

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

MIN_MODEL_ROWS = 40
MODEL_REQUIRED_COLUMNS = ["price_gbp", *MODEL_FEATURES]


def currency(value: float) -> str:
    return f"£{value:,.2f}"


@st.cache_data(show_spinner=False)
def load_books() -> pd.DataFrame:
    processed_path = Path(PROCESSED_BOOKS_PATH)
    raw_path = Path(RAW_BOOKS_PATH)

    if processed_path.exists():
        return pd.read_csv(processed_path)

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        clean_df = transform_books(raw_df)
        save_processed_books(clean_df)
        return clean_df

    return pd.DataFrame()


def collect_data(max_pages: int) -> pd.DataFrame:
    raw_df = scrape_books(
        max_pages=max_pages,
        delay=0.03,
        checkpoint_path=RAW_BOOKS_PATH,
        progress=True,
    )
    save_raw_books(raw_df)
    clean_df = transform_books(raw_df)
    save_processed_books(clean_df)
    return clean_df


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda data: str(pd.util.hash_pandas_object(data, index=True).sum())},
)
def train_ml_report(df: pd.DataFrame) -> dict[str, object]:
    return build_ml_model(df, include_estimator=True)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    max_pages = st.sidebar.slider("Paginas para nova coleta", 2, 50, 50, 1)
    if st.sidebar.button("Atualizar coleta", type="primary"):
        with st.spinner("Coletando catalogo e paginas de detalhe..."):
            collect_data(max_pages=max_pages)
        st.cache_data.clear()
        st.rerun()

    categories = sorted(df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Categorias",
        categories,
        placeholder="Todas as categorias",
    )
    selected_ratings = st.sidebar.multiselect(
        "Avaliacoes",
        [1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        format_func=lambda value: f"{value} estrela" if value == 1 else f"{value} estrelas",
    )

    min_price = float(df["price_gbp"].min())
    max_price = float(df["price_gbp"].max())
    price_range = st.sidebar.slider(
        "Faixa de preco (£)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=0.5,
    )
    only_available = st.sidebar.checkbox("Somente disponiveis", value=False)
    include_outliers = st.sidebar.checkbox("Incluir outliers", value=True)

    filtered = df[
        df["rating"].isin(selected_ratings)
        & df["price_gbp"].between(price_range[0], price_range[1])
    ].copy()

    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if only_available:
        filtered = filtered[filtered["availability_status"].eq("Disponivel")]
    if not include_outliers:
        filtered = filtered[~filtered["is_price_outlier"]]

    return filtered


def kpi_row(df: pd.DataFrame, total_rows: int) -> None:
    books, categories, median_price, mean_rating, outliers = st.columns(5)
    books.metric("Livros", f"{len(df):,}".replace(",", "."), f"{len(df) - total_rows:+}")
    categories.metric("Categorias", f"{df['category'].nunique():,}".replace(",", "."))
    median_price.metric("Preco mediano", currency(float(df["price_gbp"].median())))
    mean_rating.metric("Avaliacao media", f"{df['rating'].mean():.2f}")
    outliers.metric("Outliers", f"{int(df['is_price_outlier'].sum())}")


def analytical_takeaway(df: pd.DataFrame) -> str:
    category_counts = df["category"].value_counts()
    top_category = category_counts.index[0]
    top_share = category_counts.iloc[0] / len(df) * 100
    median_price = df["price_gbp"].median()
    high_value = df.sort_values("value_score", ascending=False).iloc[0]
    return (
        f"**Leitura principal:** {top_category} concentra {top_share:.1f}% do recorte, "
        f"com preco mediano geral de {currency(float(median_price))}. "
        f"O melhor equilibrio entre avaliacao e preco aparece em "
        f"**{high_value['title']}**, com score de valor {high_value['value_score']:.2f}."
    )


def render_overview(df: pd.DataFrame) -> None:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.plotly_chart(
            price_histogram(df),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            category_median_bar(df),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            rating_price_box(df),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            stock_price_scatter(df),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )


def render_statistics(df: pd.DataFrame) -> None:
    correlations = pearson_correlations(df)
    quartiles = quartile_summary(df)
    anova_rating = anova_by_group(df, group_col="rating_label")
    anova_category = anova_by_group(df, group_col="category")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(
            correlation_bar(correlations),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            quartile_bar(quartiles),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )

    st.subheader("Testes estatisticos")
    stat_cols = st.columns(4)
    stat_cols[0].metric("ANOVA por avaliacao", f"F={anova_rating['f_statistic']:.2f}")
    stat_cols[1].metric("p-valor", f"{anova_rating['p_value']:.4f}")
    stat_cols[2].metric("ANOVA por categoria", f"F={anova_category['f_statistic']:.2f}")
    stat_cols[3].metric("p-valor", f"{anova_category['p_value']:.4f}")

    st.dataframe(
        correlations.assign(
            correlacao_pearson=lambda data: data["correlacao_pearson"].round(4),
            p_valor=lambda data: data["p_valor"].round(4),
        ),
        use_container_width=True,
        hide_index=True,
    )

    outliers = outlier_table(df)
    st.subheader("Livros fora do padrao pelo criterio IQR")
    st.dataframe(
        outliers.assign(
            price_gbp=lambda data: data["price_gbp"].round(2),
            price_vs_category_median_pct=lambda data: data["price_vs_category_median_pct"].round(1),
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_segments(df: pd.DataFrame) -> None:
    summary = category_summary(df)
    st.dataframe(
        summary.assign(
            preco_medio=lambda data: data["preco_medio"].round(2),
            preco_mediano=lambda data: data["preco_mediano"].round(2),
            preco_minimo=lambda data: data["preco_minimo"].round(2),
            preco_maximo=lambda data: data["preco_maximo"].round(2),
            avaliacao_media=lambda data: data["avaliacao_media"].round(2),
            valor_estoque=lambda data: data["valor_estoque"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected_category = st.selectbox("Categoria em foco", summary["category"].tolist())
    focus = df[df["category"].eq(selected_category)].sort_values("price_gbp", ascending=False)
    st.dataframe(
        focus[
            [
                "title",
                "price_gbp",
                "rating",
                "stock_quantity",
                "value_score",
                "price_vs_category_median_pct",
            ]
        ].assign(
            price_gbp=lambda data: data["price_gbp"].round(2),
            value_score=lambda data: data["value_score"].round(2),
            price_vs_category_median_pct=lambda data: data["price_vs_category_median_pct"].round(1),
        ),
        use_container_width=True,
        hide_index=True,
    )


def word_count(value: str) -> int:
    return len(value.split())


def category_price_stats(df: pd.DataFrame, category: str) -> dict[str, float | int]:
    prices = df.loc[df["category"].eq(category), "price_gbp"].dropna()
    if prices.empty:
        prices = df["price_gbp"].dropna()

    return {
        "livros": int(len(prices)),
        "min": float(prices.min()),
        "q1": float(prices.quantile(0.25)),
        "median": float(prices.median()),
        "q3": float(prices.quantile(0.75)),
        "max": float(prices.max()),
    }


def reference_book_for_profile(category_df: pd.DataFrame, profile: str) -> pd.Series:
    reference_df = category_df.dropna(subset=["price_gbp"]).copy()
    if profile == "Mais barato":
        return reference_df.sort_values("price_gbp").iloc[0]
    if profile == "Mais caro":
        return reference_df.sort_values("price_gbp", ascending=False).iloc[0]
    if profile == "Melhor avaliado":
        return reference_df.sort_values(["rating", "price_gbp"], ascending=[False, True]).iloc[0]

    median_price = reference_df["price_gbp"].median()
    return (
        reference_df.assign(distance_to_median=(reference_df["price_gbp"] - median_price).abs())
        .sort_values(["distance_to_median", "rating"], ascending=[True, False])
        .iloc[0]
    )


def price_position_label(predicted_price: float, stats: dict[str, float | int]) -> tuple[str, str]:
    if predicted_price < float(stats["q1"]):
        return "Abaixo da faixa central", "competitivo"
    if predicted_price > float(stats["q3"]):
        return "Acima da faixa central", "premium"
    return "Dentro da faixa central", "alinhado"


def nearest_books(df: pd.DataFrame, category: str, predicted_price: float, limit: int = 5) -> pd.DataFrame:
    candidates = df.loc[df["category"].eq(category)].copy()
    if len(candidates) < limit:
        candidates = df.copy()

    return (
        candidates.assign(price_distance=(candidates["price_gbp"] - predicted_price).abs())
        .sort_values(["price_distance", "rating"], ascending=[True, False])
        .head(limit)
    )


def render_price_simulator(df: pd.DataFrame, model_report: dict[str, object]) -> None:
    estimator = model_report.get("estimator")
    if estimator is None:
        return

    st.subheader("Simulador de preço")
    categories = sorted(df["category"].dropna().unique())
    availability_options = sorted(df["availability_status"].dropna().unique())
    max_stock = max(100, int(df["stock_quantity"].max()) + 20)
    max_reviews = max(1000, int(df["num_reviews"].max()) + 100)

    category_col, profile_col = st.columns([1.25, 0.75])
    with category_col:
        selected_category = st.selectbox("Categoria", categories, key="ml_category")
    with profile_col:
        selected_profile = st.selectbox(
            "Perfil inicial",
            ["Mediano da categoria", "Mais barato", "Mais caro", "Melhor avaliado"],
            key="ml_profile",
        )

    category_df = df.loc[df["category"].eq(selected_category)]
    reference = reference_book_for_profile(category_df, selected_profile)
    stats = category_price_stats(df, selected_category)
    key_suffix = f"{selected_category}_{selected_profile}"

    with st.container(border=True):
        st.markdown("##### Perfil do livro")
        left, right = st.columns(2)
        with left:
            rating = st.slider(
                "Avaliação",
                1,
                5,
                int(reference["rating"]),
                key=f"ml_rating_{key_suffix}",
            )
            stock_quantity = st.number_input(
                "Estoque declarado",
                min_value=0,
                max_value=max_stock,
                value=int(reference["stock_quantity"]),
                step=1,
                key=f"ml_stock_{key_suffix}",
            )
            num_reviews = st.number_input(
                "Reviews",
                min_value=0,
                max_value=max_reviews,
                value=int(reference["num_reviews"]),
                step=1,
                key=f"ml_reviews_{key_suffix}",
            )
        with right:
            reference_availability = str(reference["availability_status"])
            default_availability = (
                availability_options.index(reference_availability)
                if reference_availability in availability_options
                else 0
            )
            availability_status = st.selectbox(
                "Disponibilidade",
                availability_options,
                index=default_availability,
                key=f"ml_availability_{key_suffix}",
            )
            title = st.text_input(
                "Título",
                value=str(reference["title"]),
                key=f"ml_title_{key_suffix}",
            )
            description = st.text_area(
                "Descrição",
                value=str(reference["description"]),
                height=118,
                key=f"ml_description_{key_suffix}",
            )

    title_words = word_count(title)
    description_words = word_count(description)
    tone_score = title_tone_score(title)
    feature_values = {
        "rating": rating,
        "stock_quantity": stock_quantity,
        "title_word_count": title_words,
        "title_char_count": len(title),
        "description_word_count": description_words,
        "title_tone_score": tone_score,
        "num_reviews": num_reviews,
        "category": selected_category,
        "availability_status": availability_status,
        "title": title,
        "description": description,
    }
    features = pd.DataFrame([feature_values])
    predicted_price = max(0.0, float(estimator.predict(features)[0]))
    mae = float(model_report.get("mae", 0.0))
    lower_bound = max(0.0, predicted_price - mae)
    upper_bound = predicted_price + mae
    category_median = float(stats["median"])
    delta = predicted_price - category_median
    delta_pct = (delta / category_median) * 100 if category_median else 0.0
    position_label, position_type = price_position_label(predicted_price, stats)
    percentile = float(category_df["price_gbp"].dropna().le(predicted_price).mean() * 100)

    with st.container(border=True):
        st.markdown("##### Resultado estimado")
        predicted, interval, difference = st.columns(3)
        predicted.metric("Preço estimado", currency(predicted_price))
        interval.metric("Faixa pelo MAE", f"{currency(lower_bound)} a {currency(upper_bound)}")
        difference.metric("Vs. mediana da categoria", currency(delta), f"{delta_pct:+.1f}%")

        st.progress(
            min(max(percentile / 100, 0.0), 1.0),
            text=f"Percentil estimado na categoria: {percentile:.0f} de 100",
        )
        st.info(
            f"{position_label}: preço {position_type} frente à faixa central da categoria "
            f"({currency(float(stats['q1']))} a {currency(float(stats['q3']))})."
        )

        derived_left, derived_middle, derived_right = st.columns(3)
        derived_left.metric("Palavras no título", title_words)
        derived_middle.metric("Palavras na descrição", description_words)
        derived_right.metric("Tom do título", f"{tone_score:+d}")

    observed_stock_min = int(df["stock_quantity"].min())
    observed_stock_max = int(df["stock_quantity"].max())
    stock_low = feature_values | {"stock_quantity": observed_stock_min}
    stock_high = feature_values | {"stock_quantity": observed_stock_max}
    rating_low = feature_values | {"rating": 1}
    rating_high = feature_values | {"rating": 5}
    stock_effect = float(
        estimator.predict(pd.DataFrame([stock_high]))[0]
        - estimator.predict(pd.DataFrame([stock_low]))[0]
    )
    rating_effect = float(
        estimator.predict(pd.DataFrame([rating_high]))[0]
        - estimator.predict(pd.DataFrame([rating_low]))[0]
    )

    with st.expander("Sensibilidade da previsão", expanded=False):
        sensitivity_left, sensitivity_right = st.columns(2)
        sensitivity_left.metric(
            f"Estoque {observed_stock_min} -> {observed_stock_max}",
            f"£{stock_effect:+.2f}",
        )
        sensitivity_right.metric("Avaliação 1 -> 5", f"£{rating_effect:+.2f}")
        if abs(stock_effect) < 1:
            st.caption(
                "O estoque tem baixo impacto neste modelo porque, nos dados coletados, "
                "a relação entre estoque declarado e preço é praticamente nula."
            )

    st.markdown("##### Referências próximas no catálogo")
    similar = nearest_books(df, selected_category, predicted_price)
    similar_display = similar[
        ["title", "price_gbp", "rating", "stock_quantity", "availability_status"]
    ].rename(
        columns={
            "title": "Título",
            "price_gbp": "Preço (£)",
            "rating": "Avaliação",
            "stock_quantity": "Estoque",
            "availability_status": "Disponibilidade",
        }
    )
    st.dataframe(
        similar_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Preço (£)": st.column_config.NumberColumn("Preço (£)", format="£%.2f"),
            "Avaliação": st.column_config.NumberColumn("Avaliação", format="%d"),
            "Estoque": st.column_config.NumberColumn("Estoque", format="%d"),
        },
    )


def count_ml_ready_rows(df: pd.DataFrame) -> int:
    missing_columns = [column for column in MODEL_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        return 0
    return int(df[MODEL_REQUIRED_COLUMNS].dropna().shape[0])


def select_model_training_data(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    filtered_rows = count_ml_ready_rows(filtered_df)
    full_rows = count_ml_ready_rows(full_df)

    if filtered_rows >= MIN_MODEL_ROWS:
        return filtered_df

    if full_rows >= MIN_MODEL_ROWS:
        st.info(
            f"O recorte atual tem {filtered_rows} livros válidos para ML. "
            f"Para manter o simulador disponível, o modelo foi treinado com "
            f"o dataset completo ({full_rows} livros)."
        )
        return full_df

    return filtered_df


def render_model(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    training_df = select_model_training_data(filtered_df, full_df)
    model_report = train_ml_report(training_df)
    if not model_report.get("available"):
        st.warning(str(model_report.get("reason", "Modelo indisponível.")))
        return

    model, train_rows, test_rows, mae, r2 = st.columns(5)
    model.metric("Modelo ML do simulador", str(model_report["algorithm"]))
    train_rows.metric("Treino", f"{int(model_report['train_rows']):,}".replace(",", "."))
    test_rows.metric("Teste", f"{int(model_report['test_rows']):,}".replace(",", "."))
    mae.metric("MAE teste", currency(float(model_report["mae"])))
    r2.metric("R² teste", f"{float(model_report['r2']):.3f}")
    st.caption(
        "Regressão supervisionada treinada com os dados processados do catálogo. "
        "Entradas: categoria, avaliação, estoque, disponibilidade, reviews e métricas "
        "do título/descrição, além de TF-IDF do texto do título e da descrição. "
        "Alvo previsto: preço do livro (`price_gbp`)."
    )

    left, right = st.columns([0.95, 1.05])
    with left:
        st.plotly_chart(
            model_mae_bar(model_report["model_metrics"]),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            predicted_vs_actual_scatter(model_report["residuals"]),
            use_container_width=True,
            theme=None,
            config=PLOT_CONFIG,
        )

    st.subheader("Comparação de modelos de machine learning")
    model_metrics = model_report["model_metrics"].assign(
        uso=lambda data: data.apply(
            lambda row: (
                "Usado no simulador"
                if row["selecionado"]
                else "Modelo ML comparado"
            ),
            axis=1,
        )
    )
    st.dataframe(
        model_metrics.drop(columns=["selecionado"]).assign(
            mae_cv=lambda data: data["mae_cv"].round(2),
            rmse_cv=lambda data: data["rmse_cv"].round(2),
            r2_cv=lambda data: data["r2_cv"].round(3),
            mae_teste=lambda data: data["mae_teste"].round(2),
            rmse_teste=lambda data: data["rmse_teste"].round(2),
            r2_teste=lambda data: data["r2_teste"].round(3),
        ),
        use_container_width=True,
        hide_index=True,
    )

    render_price_simulator(training_df, model_report)

    st.subheader("Impacto das variáveis")
    st.plotly_chart(
        feature_importance_bar(model_report["feature_importance"]),
        use_container_width=True,
        theme=None,
        config=PLOT_CONFIG,
    )

    st.subheader("Maiores erros no teste")
    st.dataframe(
        model_report["residuals"]
        .head(10)
        .assign(
            preco_real=lambda data: data["preco_real"].round(2),
            preco_previsto=lambda data: data["preco_previsto"].round(2),
            erro=lambda data: data["erro"].round(2),
            erro_absoluto=lambda data: data["erro_absoluto"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_data(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv,
        file_name="books_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Radar Editorial")
    st.caption("Scraping, ETL, estatistica e machine learning sobre precos de livros.")

    df = load_books()
    if df.empty:
        st.info("Nenhum dataset encontrado. Execute a coleta pela barra lateral.")
        with st.sidebar:
            st.header("Coleta inicial")
            max_pages = st.slider("Paginas", 2, 50, 50, 1)
            if st.button("Executar coleta", type="primary"):
                with st.spinner("Coletando dados..."):
                    collect_data(max_pages=max_pages)
                st.cache_data.clear()
                st.rerun()
        return

    filtered = sidebar_filters(df)
    if filtered.empty:
        st.warning("Nenhum livro encontrado para os filtros atuais.")
        return

    kpi_row(filtered, total_rows=len(df))
    st.markdown(analytical_takeaway(filtered))

    overview, statistics, segments, model, data = st.tabs(
        ["Panorama", "Estatistica", "Segmentos", "Modelo ML", "Dados"]
    )
    with overview:
        render_overview(filtered)
    with statistics:
        render_statistics(filtered)
    with segments:
        render_segments(filtered)
    with model:
        render_model(filtered, df)
    with data:
        render_data(filtered)


if __name__ == "__main__":
    main()
