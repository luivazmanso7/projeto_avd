# Radar Editorial - Projeto Final AVD

Dashboard em Streamlit para analisar precos, avaliacoes, estoque e categorias de livros coletados por web scraping no site [Books to Scrape](https://books.toscrape.com/).

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --max-pages 20 --delay 0.03
streamlit run app.py
```

Se o CSV limpo ja existir em `data/processed/books_clean.csv`, o dashboard abre sem nova coleta. Para capturar o catalogo inteiro, use `--max-pages 50`.

## O que foi implementado

- **Fase 1 - Web scraping:** `src/scraper.py` coleta paginas de catalogo e paginas de detalhe, lendo tags HTML de titulo, preco, avaliacao, disponibilidade, categoria, descricao e tabela tecnica.
- **Fase 2 - ETL e data wrangling:** `src/etl.py` remove duplicidades, trata nulos, converte tipos, normaliza textos e cria metricas como estoque declarado, score de valor, quartis, outliers IQR e desvio frente a mediana da categoria.
- **Fase 3 - Analise estatistica:** `src/analysis.py` calcula correlacao de Pearson, ANOVA por avaliacao e categoria, quartis e deteccao de outliers.
- **Fase 4 - Visualizacao e Gestalt:** `src/visuals.py` usa Plotly com hierarquia visual, ordenacao por relevancia, contraste controlado, cores de destaque e agrupamentos perceptivos.
- **Fase 5 - Dashboard Streamlit:** `app.py` oferece filtros de categoria, avaliacao, faixa de preco, disponibilidade e outliers, com abas para panorama, estatistica, segmentos, modelo e dados.
- **Bonus - Machine Learning:** modelo `RandomForestRegressor` estima preco dos livros e exibe MAE, R2 e importancia das variaveis.

## Estrutura

```text
.
├── app.py
├── requirements.txt
├── src/
│   ├── analysis.py
│   ├── config.py
│   ├── etl.py
│   ├── pipeline.py
│   ├── scraper.py
│   └── visuals.py
└── data/
    ├── raw/
    └── processed/
```

## Fonte dos dados

Books to Scrape e um ambiente estatico criado para pratica de scraping. O projeto usa 20 paginas por padrao para uma execucao mais rapida e permite ate 50 paginas de catalogo, totalizando 1000 livros quando a coleta completa e executada.

## Criterios da rubrica

| Criterio | Evidencia no projeto |
| --- | --- |
| Engenharia de Dados | Coleta reproduzivel, ETL modular, conversao de tipos, deduplicacao e dados cacheados |
| Qualidade Analitica | Pearson, ANOVA, quartis, IQR e interpretacao por segmento |
| Design e Gestalt | Graficos ordenados, contraste sem excesso, foco visual nos dados principais |
| Implementacao Streamlit | Dashboard interativo, filtros, abas analiticas e exportacao CSV |
| Bonus | Regressao com Random Forest e importancia de variaveis |
