# Radar Editorial - Projeto Final AVD

Dashboard em Streamlit para analisar precos, avaliacoes, estoque e categorias de livros coletados por web scraping no site [Books to Scrape](https://books.toscrape.com/).

## Como executar

### 1. Pre-requisitos

Antes de comecar, confirme que voce tem o Python instalado:

```bash
python3 --version
```

Recomendacao: use Python 3.10 ou superior.

### 2. Acessar a pasta do projeto

No terminal, entre na pasta onde o projeto foi baixado:

```bash
cd projeto_avd
```

### 3. Criar e ativar o ambiente virtual

Crie um ambiente virtual para instalar as dependencias do projeto sem afetar o Python global:

```bash
python3 -m venv .venv
```

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

No Windows, o comando equivalente e:

```bash
.venv\Scripts\activate
```

### 4. Instalar as dependencias

Com o ambiente virtual ativo, instale as bibliotecas usadas no projeto:

```bash
pip install -r requirements.txt
```

### 5. Gerar ou atualizar os dados

O projeto ja possui arquivos CSV em `data/raw/` e `data/processed/`. Se `data/processed/books_clean.csv` existir, o dashboard consegue abrir diretamente sem executar uma nova coleta.

Para refazer a coleta, o tratamento dos dados e o relatorio de analise, execute:

```bash
python -m src.pipeline --max-pages 20 --delay 0.03
```

Esse comando cria ou atualiza:

- `data/raw/books_raw.csv`
- `data/processed/books_clean.csv`
- `data/processed/analysis_report.json`

Para coletar o catalogo completo do Books to Scrape, use 50 paginas:

```bash
python -m src.pipeline --max-pages 50 --delay 0.03
```

### 6. Rodar o dashboard

Inicie a aplicacao Streamlit:

```bash
streamlit run app.py
```

Se o comando `streamlit` nao for reconhecido, use:

```bash
python -m streamlit run app.py
```

Depois disso, o Streamlit mostrara no terminal um endereco local parecido com:

```text
http://localhost:8501
```

Abra esse endereco no navegador para acessar o dashboard.

### 7. Atualizar a coleta pelo dashboard

Com o dashboard aberto, tambem e possivel atualizar os dados pela barra lateral:

1. Escolha a quantidade de paginas em **Paginas para nova coleta**.
2. Clique em **Atualizar coleta**.
3. Aguarde a coleta terminar e a pagina recarregar.

### 8. Encerrar a aplicacao

Para parar o servidor do Streamlit, volte ao terminal onde ele esta rodando e pressione:

```text
Ctrl + C
```

### Problemas comuns

- **Dashboard abriu sem dados:** execute `python -m src.pipeline --max-pages 20 --delay 0.03` e rode o dashboard novamente.
- **Erro ao coletar dados:** confira se ha conexao com a internet, pois a coleta acessa `https://books.toscrape.com/`.
- **Dependencia nao encontrada:** confirme que o ambiente virtual esta ativo e rode novamente `pip install -r requirements.txt`.

## O que foi implementado

- **Fase 1 - Web scraping:** `src/scraper.py` coleta paginas de catalogo e paginas de detalhe, lendo tags HTML de titulo, preco, avaliacao, disponibilidade, categoria, descricao e tabela tecnica.
- **Fase 2 - ETL e data wrangling:** `src/etl.py` remove duplicidades, trata nulos, converte tipos, normaliza textos e cria metricas como estoque declarado, score de valor, quartis, outliers IQR e desvio frente a mediana da categoria.
- **Fase 3 - Analise estatistica:** `src/analysis.py` calcula correlacao de Pearson, ANOVA por avaliacao e categoria, quartis e deteccao de outliers.
- **Fase 4 - Visualizacao e Gestalt:** `src/visuals.py` usa Plotly com hierarquia visual, ordenacao por relevancia, contraste controlado, cores de destaque e agrupamentos perceptivos.
- **Fase 5 - Dashboard Streamlit:** `app.py` oferece filtros de categoria, avaliacao, faixa de preco, disponibilidade e outliers, com abas para panorama, estatistica, segmentos, modelo e dados.
- **Bonus - Machine Learning:** compara baseline, Ridge, Random Forest e Gradient Boosting para prever preco, usando validacao cruzada, teste holdout, MAE, RMSE, R2, importancia por permutacao, analise de residuos e simulador de preco.

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
| Bonus | Comparacao de modelos, baseline, validacao cruzada, metricas de teste, importancia por permutacao e simulador preditivo |
