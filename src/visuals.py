from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORWAY = ["#1B7F78", "#D4553A", "#2F6F9F", "#B88400", "#6A4C93", "#B65F00"]
INK = "#1F2933"
MUTED = "#46556B"
GRID = "#D0D5DD"
ACCENT = "#D4553A"
POSITIVE = "#1B7F78"
NEGATIVE = "#C44536"
PANEL = "#FFFFFF"

RATING_COLORS = {
    "1 estrela": "#9B5DE5",
    "2 estrelas": "#0077B6",
    "3 estrelas": "#F8961E",
    "4 estrelas": "#D62828",
    "5 estrelas": "#1B9E77",
}

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
}


def apply_layout(fig: go.Figure, title: str | None = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": INK},
        }
        if title
        else None,
        template="plotly_white",
        height=height,
        colorway=COLORWAY,
        font={
            "family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
            "color": INK,
            "size": 14,
        },
        margin={"l": 72, "r": 28, "t": 78 if title else 30, "b": 78},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13, "color": INK},
            "title": {"font": {"size": 13, "color": MUTED}},
            "bgcolor": "rgba(255,255,255,0.92)",
        },
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#D0D5DD",
            "font": {"color": INK, "size": 13},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
        linewidth=1,
        tickcolor=GRID,
        tickfont={"color": INK, "size": 13},
        title_font={"color": MUTED, "size": 16},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        linewidth=1,
        tickcolor=GRID,
        tickfont={"color": INK, "size": 13},
        title_font={"color": MUTED, "size": 16},
    )
    return fig


def price_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="price_gbp",
        nbins=24,
        color_discrete_sequence=[ACCENT],
        labels={"price_gbp": "Preço (£)", "count": "Livros"},
    )
    median = df["price_gbp"].median()
    fig.add_vline(
        x=median,
        line_dash="dash",
        line_color=INK,
        annotation_text=f"Mediana £{median:.2f}",
        annotation_position="top right",
        annotation_font_color=INK,
        annotation_font_size=14,
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return apply_layout(fig, "Distribuição de preços")


def category_median_bar(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    summary = (
        df.groupby("category")
        .agg(livros=("title", "count"), preco_mediano=("price_gbp", "median"))
        .sort_values("livros", ascending=False)
        .head(top_n)
        .sort_values("preco_mediano")
        .reset_index()
    )
    highlight = summary["preco_mediano"].eq(summary["preco_mediano"].max())
    colors = np.where(highlight, ACCENT, "#2F6F9F")
    fig = go.Figure(
        go.Bar(
            x=summary["preco_mediano"],
            y=summary["category"],
            orientation="h",
            marker_color=colors,
            text=summary["preco_mediano"].map(lambda value: f"£{value:.1f}"),
            textposition="outside",
            cliponaxis=False,
            customdata=summary[["livros"]],
            hovertemplate=(
                "%{y}<br>Preço mediano: £%{x:.2f}<br>"
                "Livros: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Preço mediano (£)")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    fig = apply_layout(fig, "Categorias com maior presença no catálogo", height=500)
    fig.update_layout(margin={"l": 150, "r": 58, "t": 78, "b": 70})
    return fig


def rating_price_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="rating_label",
        y="price_gbp",
        color="rating_label",
        color_discrete_map=RATING_COLORS,
        points="outliers",
        labels={"rating_label": "Avaliação", "price_gbp": "Preço (£)"},
        category_orders={
            "rating_label": [
                "1 estrela",
                "2 estrelas",
                "3 estrelas",
                "4 estrelas",
                "5 estrelas",
            ]
        },
    )
    fig.update_traces(boxmean=True, line_width=2.4, marker={"size": 5, "opacity": 0.75})
    fig.update_layout(showlegend=False)
    return apply_layout(fig, "Variação de preço por avaliação")


def stock_price_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    rng = np.random.default_rng(42)
    plot_df["stock_jittered"] = plot_df["stock_quantity"] + rng.normal(0, 0.09, len(plot_df))

    fig = px.scatter(
        plot_df,
        x="stock_jittered",
        y="price_gbp",
        color="rating_label",
        color_discrete_map=RATING_COLORS,
        size="stock_quantity",
        size_max=14,
        hover_name="title",
        hover_data={
            "category": True,
            "stock_quantity": True,
            "stock_jittered": False,
            "price_gbp": ":.2f",
            "rating_label": True,
            "value_score": ":.2f",
        },
        labels={
            "stock_quantity": "Estoque declarado",
            "price_gbp": "Preço (£)",
            "rating_label": "Avaliação",
        },
    )
    fig.update_traces(marker={"opacity": 0.62, "line": {"width": 0.8, "color": "#FFFFFF"}})
    fig.update_xaxes(title="Estoque declarado", tickmode="linear", dtick=2)
    return apply_layout(fig, "Preço, estoque e avaliação")


def correlation_bar(correlations: pd.DataFrame) -> go.Figure:
    if correlations.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Amostra insuficiente para correlação",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": MUTED, "size": 15},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_layout(fig, "Correlação de Pearson com preço")

    plot_df = correlations.copy()
    plot_df["variavel_label"] = plot_df["variavel"].map(FEATURE_LABELS).fillna(plot_df["variavel"])
    plot_df["correlacao_abs"] = plot_df["correlacao_pearson"].abs()
    plot_df = plot_df.sort_values("correlacao_abs")
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in plot_df["correlacao_pearson"]]
    axis_limit = max(0.05, float(plot_df["correlacao_abs"].max()) * 1.25)
    fig = go.Figure(
        go.Bar(
            x=plot_df["correlacao_pearson"],
            y=plot_df["variavel_label"],
            orientation="h",
            marker_color=colors,
            customdata=plot_df[["p_valor", "forca"]],
            text=plot_df["correlacao_pearson"].map(lambda value: f"{value:+.3f}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Correlação: %{x:.3f}<br>"
                "p-valor: %{customdata[0]:.4f}<br>"
                "Força: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(
        title="Correlação de Pearson (r)",
        range=[-axis_limit, axis_limit],
        tickformat=".2f",
    )
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    fig = apply_layout(fig, "Correlação de Pearson com preço")
    fig.update_layout(margin={"l": 185, "r": 72, "t": 78, "b": 72})
    return fig


def quartile_bar(quartiles: pd.DataFrame) -> go.Figure:
    plot_df = quartiles.copy()
    colors = ["#2A9D8F", "#A8B86F", "#E9A857", "#E76F51"][: len(plot_df)]
    fig = go.Figure(
        go.Bar(
            x=plot_df["price_quartile"],
            y=plot_df["preco_mediano"],
            marker_color=colors,
            text=plot_df["preco_mediano"].map(lambda value: f"£{value:.2f}"),
            textposition="outside",
            cliponaxis=False,
            customdata=plot_df[["preco_minimo", "preco_maximo", "livros"]],
            hovertemplate=(
                "%{x}<br>"
                "Mediana: £%{y:.2f}<br>"
                "Faixa: £%{customdata[0]:.2f} a £%{customdata[1]:.2f}<br>"
                "Livros: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Preço mediano (£)", tickprefix="£")
    fig.update_xaxes(title="", tickangle=-18)
    fig = apply_layout(fig, "Faixas de preço por quartil")
    fig.update_layout(margin={"l": 86, "r": 44, "t": 78, "b": 118})
    return fig


def feature_importance_bar(importance: pd.DataFrame) -> go.Figure:
    plot_df = importance.sort_values("importance")
    fig = px.bar(
        plot_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#DDE7EE", "#2A9D8F", "#E76F51"],
        labels={"importance": "Impacto no MAE", "feature": ""},
    )
    fig.update_coloraxes(showscale=False)
    fig = apply_layout(fig, "Impacto das variáveis no erro")
    fig.update_layout(margin={"l": 230, "r": 34, "t": 78, "b": 72})
    return fig


def model_mae_bar(metrics: pd.DataFrame) -> go.Figure:
    plot_df = metrics.sort_values("mae_teste", ascending=False)
    colors = np.where(plot_df["selecionado"], ACCENT, "#2F6F9F")
    fig = go.Figure(
        go.Bar(
            x=plot_df["mae_teste"],
            y=plot_df["modelo"],
            orientation="h",
            marker_color=colors,
            customdata=plot_df[["mae_cv", "rmse_teste", "r2_teste"]],
            hovertemplate=(
                "MAE teste: £%{x:.2f}<br>"
                "MAE CV: £%{customdata[0]:.2f}<br>"
                "RMSE teste: £%{customdata[1]:.2f}<br>"
                "R2 teste: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Erro absoluto médio no teste (£)")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    fig = apply_layout(fig, "Comparação de modelos de ML", height=380)
    fig.update_layout(margin={"l": 170, "r": 34, "t": 78, "b": 72})
    return fig


def predicted_vs_actual_scatter(residuals: pd.DataFrame) -> go.Figure:
    if residuals.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sem previsões de teste disponíveis",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": MUTED, "size": 15},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_layout(fig, "Preço real vs previsto")

    limit_min = min(residuals["preco_real"].min(), residuals["preco_previsto"].min())
    limit_max = max(residuals["preco_real"].max(), residuals["preco_previsto"].max())
    fig = px.scatter(
        residuals,
        x="preco_real",
        y="preco_previsto",
        color="erro_absoluto",
        color_continuous_scale=["#2A9D8F", "#E9C46A", "#E76F51"],
        hover_name="title",
        hover_data={
            "category": True,
            "preco_real": ":.2f",
            "preco_previsto": ":.2f",
            "erro": ":.2f",
            "erro_absoluto": ":.2f",
        },
        labels={
            "preco_real": "Preço real (£)",
            "preco_previsto": "Preço previsto (£)",
            "erro_absoluto": "Erro absoluto",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[limit_min, limit_max],
            y=[limit_min, limit_max],
            mode="lines",
            line={"color": INK, "dash": "dash", "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_coloraxes(showscale=False)
    fig = apply_layout(fig, "Preço real vs previsto")
    fig.update_layout(margin={"l": 82, "r": 34, "t": 78, "b": 72})
    return fig
