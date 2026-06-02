from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import CATALOGUE_URL, RAW_BOOKS_PATH


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    )
}


def fetch_html(
    url: str,
    timeout: int = 20,
    session: requests.Session | None = None,
    retries: int = 3,
    backoff: float = 0.5,
) -> str:
    client = session or requests
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(backoff * attempt)

    raise RuntimeError(f"Falha ao baixar {url}: {last_error}") from last_error


def parse_rating(article: BeautifulSoup) -> str | None:
    rating_tag = article.select_one("p.star-rating")
    if not rating_tag:
        return None
    for class_name in rating_tag.get("class", []):
        if class_name != "star-rating":
            return class_name
    return None


def parse_catalog_page(html: str, page_url: str) -> list[dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    books: list[dict[str, str | None]] = []

    for article in soup.select("article.product_pod"):
        title_tag = article.select_one("h3 a")
        price_tag = article.select_one(".price_color")
        availability_tag = article.select_one(".availability")
        image_tag = article.select_one(".image_container img")

        if not title_tag:
            continue

        detail_url = urljoin(page_url, title_tag.get("href", ""))
        image_url = urljoin(page_url, image_tag.get("src", "")) if image_tag else None

        books.append(
            {
                "title": title_tag.get("title", title_tag.get_text(strip=True)),
                "catalogue_price": price_tag.get_text(strip=True) if price_tag else None,
                "rating_text": parse_rating(article),
                "catalogue_availability": (
                    availability_tag.get_text(" ", strip=True) if availability_tag else None
                ),
                "detail_url": detail_url,
                "image_url": image_url,
            }
        )

    return books


def table_to_dict(soup: BeautifulSoup) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in soup.select("table.table-striped tr"):
        header = row.find("th")
        value = row.find("td")
        if header and value:
            values[header.get_text(" ", strip=True)] = value.get_text(" ", strip=True)
    return values


def parse_product_page(html: str) -> dict[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    product_table = table_to_dict(soup)

    breadcrumbs = [item.get_text(" ", strip=True) for item in soup.select("ul.breadcrumb li")]
    category = breadcrumbs[2] if len(breadcrumbs) > 2 else None

    description_header = soup.select_one("#product_description")
    description = None
    if description_header:
        description_tag = description_header.find_next_sibling("p")
        description = description_tag.get_text(" ", strip=True) if description_tag else None

    return {
        "category": category,
        "description": description,
        "upc": product_table.get("UPC"),
        "product_type": product_table.get("Product Type"),
        "price_excl_tax": product_table.get("Price (excl. tax)"),
        "price_incl_tax": product_table.get("Price (incl. tax)"),
        "tax": product_table.get("Tax"),
        "availability_detail": product_table.get("Availability"),
        "num_reviews": product_table.get("Number of reviews"),
    }


def scrape_books(
    max_pages: int = 20,
    delay: float = 0.1,
    timeout: int = 20,
    checkpoint_path: Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Scrape the Books to Scrape catalogue and product detail pages."""
    records: list[dict[str, str | None]] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, max_pages + 1):
        page_url = CATALOGUE_URL.format(page=page)
        if progress:
            print(f"Coletando pagina {page}/{max_pages}: {page_url}", flush=True)

        html = fetch_html(page_url, timeout=timeout, session=session)
        catalog_books = parse_catalog_page(html, page_url)
        if not catalog_books:
            break

        for book in catalog_books:
            detail_url = str(book["detail_url"])
            detail_html = fetch_html(detail_url, timeout=timeout, session=session)
            book.update(parse_product_page(detail_html))
            records.append(book)
            if delay > 0:
                time.sleep(delay)

        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(checkpoint_path, index=False)

        if delay > 0:
            time.sleep(delay)

    return pd.DataFrame(records)


def save_raw_books(df: pd.DataFrame, output_path: Path = RAW_BOOKS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coleta dados do Books to Scrape.")
    parser.add_argument("--max-pages", type=int, default=20, help="Quantidade maxima de paginas.")
    parser.add_argument("--delay", type=float, default=0.1, help="Intervalo entre requisicoes.")
    parser.add_argument("--output", type=Path, default=RAW_BOOKS_PATH, help="CSV de saida.")
    parser.add_argument("--progress", action="store_true", help="Mostra progresso por pagina.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = scrape_books(
        max_pages=args.max_pages,
        delay=args.delay,
        checkpoint_path=args.output,
        progress=args.progress,
    )
    output_path = save_raw_books(df, args.output)
    print(f"Coleta concluida: {len(df)} livros salvos em {output_path}")


if __name__ == "__main__":
    main()
