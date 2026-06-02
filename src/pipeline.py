from __future__ import annotations

import argparse
from typing import Iterable

from .analysis import build_analysis_report, save_analysis_report
from .config import RAW_BOOKS_PATH
from .etl import save_processed_books, transform_books
from .scraper import save_raw_books, scrape_books


def run_pipeline(max_pages: int = 20, delay: float = 0.05) -> dict[str, int | str]:
    raw_df = scrape_books(
        max_pages=max_pages,
        delay=delay,
        checkpoint_path=RAW_BOOKS_PATH,
        progress=True,
    )
    raw_path = save_raw_books(raw_df)

    clean_df = transform_books(raw_df)
    clean_path = save_processed_books(clean_df)

    report = build_analysis_report(clean_df)
    report_path = save_analysis_report(report)

    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "report_path": str(report_path),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa scraping, ETL, analise e ML.")
    parser.add_argument("--max-pages", type=int, default=20, help="Quantidade maxima de paginas.")
    parser.add_argument("--delay", type=float, default=0.05, help="Intervalo entre requisicoes.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_pipeline(max_pages=args.max_pages, delay=args.delay)
    print(
        "Pipeline concluido: "
        f"{result['raw_rows']} linhas brutas, {result['clean_rows']} linhas limpas. "
        f"Arquivos: {result['raw_path']}, {result['clean_path']}, {result['report_path']}"
    )


if __name__ == "__main__":
    main()
