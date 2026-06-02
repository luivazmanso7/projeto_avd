from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import (
    anova_by_group,
    build_ml_model,
    category_summary,
    outlier_table,
    pearson_correlations,
    quartile_summary,
)
from src.config import PROCESSED_BOOKS_PATH, RAW_BOOKS_PATH
from src.etl import save_processed_books, transform_books
from src.scraper import save_raw_books, scrape_books
from src.visuals import (
    category_median_bar,
    correlation_bar,
    feature_importance_bar,
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


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    max_pages = st.sidebar.slider("Paginas para nova coleta", 1, 50, 20, 1)
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


def render_model(df: pd.DataFrame) -> None:
    model_report = build_ml_model(df)
    if not model_report.get("available"):
        st.warning(str(model_report.get("reason", "Modelo indisponivel.")))
        return

    left, middle, right = st.columns(3)
    left.metric("Algoritmo", str(model_report["algorithm"]))
    middle.metric("MAE", currency(float(model_report["mae"])))
    right.metric("R2", f"{float(model_report['r2']):.3f}")

    importance = model_report["feature_importance"]
    st.plotly_chart(
        feature_importance_bar(importance),
        use_container_width=True,
        theme=None,
        config=PLOT_CONFIG,
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
            max_pages = st.slider("Paginas", 1, 50, 20, 1)
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
        render_model(filtered)
    with data:
        render_data(filtered)


if __name__ == "__main__":
    main()
