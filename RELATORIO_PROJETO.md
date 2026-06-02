# Relatorio Academico - Radar Editorial

## Objetivo

Analisar o comportamento de precos de livros em um catalogo de e-commerce, observando como categoria, avaliacao, estoque e sinais textuais se relacionam com o preco final.

## Metodologia

1. **Extracao:** coleta de HTML no Books to Scrape com `requests` e `BeautifulSoup`, visitando paginas de catalogo e detalhe.
2. **ETL:** padronizacao de campos, tratamento de nulos, conversao de precos para numerico, criacao de metricas calculadas e marcacao de outliers.
3. **Estatistica:** correlacao de Pearson entre preco e variaveis numericas, ANOVA para diferencas entre grupos, quartis e IQR.
4. **Visualizacao:** graficos com ordenacao intencional, destaque cromatico moderado, comparacao por proximidade e consistencia de codificacao visual.
5. **Aplicacao:** dashboard Streamlit com filtros e abas analiticas.
6. **Inovacao:** Random Forest para estimar preco e indicar importancia das variaveis.

## Dicionario de variaveis principais

| Variavel | Descricao |
| --- | --- |
| `price_gbp` | Preco final em libra esterlina |
| `rating` | Avaliacao convertida para escala de 1 a 5 |
| `stock_quantity` | Quantidade disponivel declarada na pagina |
| `value_score` | Razao entre avaliacao e preco, multiplicada por 100 |
| `price_quartile` | Quartil de preco no catalogo |
| `is_price_outlier` | Indicador de outlier pelo metodo IQR global ou por categoria |
| `price_vs_category_median_pct` | Diferenca percentual frente a mediana da categoria |

## Decisoes de design visual

- Barras horizontais para categorias, favorecendo leitura de rotulos longos.
- Histograma com linha de mediana para orientar a distribuicao de preco.
- Cores contrastantes apenas quando diferenciam grupos ou direcao de efeito.
- Tabelas de segmento para detalhamento sem poluir os graficos principais.

## Limitacoes

O Books to Scrape e uma base de treinamento, portanto nao representa um mercado real em tempo corrente. A analise e adequada para demonstrar ciclo tecnico, estatistica e visualizacao, mas nao deve ser usada como decisao comercial real.
