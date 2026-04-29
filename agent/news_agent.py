import feedparser
import html
import requests
import re
import logging
from urllib.parse import urlparse

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from newspaper import Article
except Exception:
    Article = None

try:
    from readability import Document as ReadabilityDocument
except Exception:
    ReadabilityDocument = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

DEFAULT_NEWS_SEARCH_QUERIES = [
    "금리 대출",
    "대출 규제",
    "카드론",
    "신용대출",
    "DSR",
    "연체",
    "사업자대출",
    "저축은행 대출",
]

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _extract_first_href(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'href=["\']([^"\']+)["\']', text, flags=re.I)
    if not match:
        return ""
    return html.unescape(match.group(1)).strip()


def _looks_like_google_news(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    return "news.google.com" in host or host.endswith("google.com")


def _extract_canonical_url(html_text: str) -> str:
    if not html_text:
        return ""
    patterns = [
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return ""


def _resolve_article_url(url: str, summary: str = "", timeout: int = 8) -> tuple[str, str]:
    candidate = _extract_first_href(summary)
    if candidate and not _looks_like_google_news(candidate):
        url = candidate

    if not url:
        return "", ""

    try:
        resp = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS, allow_redirects=True)
        final_url = resp.url or url
        html_text = resp.text or ""

        if _looks_like_google_news(final_url):
            canonical = _extract_canonical_url(html_text)
            if canonical and not _looks_like_google_news(canonical):
                return canonical, html_text
        return final_url, html_text
    except Exception:
        return url, ""


def _clean_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fetch_article_text(url: str, timeout: int = 8) -> str:
    """Try to fetch and extract main article text from the given URL.

    Attempts: readability -> BeautifulSoup heuristics -> simple <p> extraction.
    Returns empty string on failure.
    """
    if not url:
        return ""
    try:
        resolved_url, prefetched_html = _resolve_article_url(url, timeout=timeout)
        if not resolved_url:
            return ""
        html = prefetched_html
        if not html:
            resp = requests.get(resolved_url, timeout=timeout, headers=_BROWSER_HEADERS)
            if resp.status_code != 200:
                return ""
            html = resp.text

        if Article is not None:
            try:
                article = Article(resolved_url, language="ko")
                article.set_html(html)
                article.parse()
                extracted = _clean_text(article.text or "")
                if len(extracted) > 200:
                    return extracted
            except Exception:
                pass

        if trafilatura is not None:
            try:
                extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
                extracted = _clean_text(extracted or "")
                if len(extracted) > 200:
                    return extracted
            except Exception:
                pass

        # readability-lxml if available
        if ReadabilityDocument is not None:
            try:
                doc = ReadabilityDocument(html)
                content = doc.summary()
                # strip tags for storage
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(content, "lxml")
                    text = soup.get_text(separator="\n\n").strip()
                    text = _clean_text(text)
                    if len(text) > 200:
                        return text
                # fallback: remove tags
                text = re.sub(r"<[^>]+>", "", content).strip()
                text = _clean_text(text)
                if len(text) > 200:
                    return text
            except Exception:
                pass

        # BeautifulSoup extraction heuristics
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "lxml")
                # prefer <article>
                article = soup.find("article")
                if article:
                    text = article.get_text(separator="\n\n").strip()
                    text = _clean_text(text)
                    if len(text) > 200:
                        return text
                # otherwise gather long <p> blocks
                ps = soup.find_all("p")
                texts = [p.get_text().strip() for p in ps if p.get_text(strip=True)]
                # heuristics: join paragraphs, choose long text
                full = "\n\n".join(texts)
                full = _clean_text(full)
                if len(full) > 200:
                    return full.strip()
            except Exception:
                pass

        # last-resort: regex extract <p>...</p>
        try:
            parts = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.S | re.I)
            cleaned = []
            for p in parts:
                t = re.sub(r"<[^>]+>", "", p).strip()
                if t:
                    cleaned.append(t)
            full = "\n\n".join(cleaned)
            full = _clean_text(full)
            if len(full) > 80:
                return full.strip()
        except Exception:
            pass

    except Exception as e:
        logger.debug("fetch_article_text failed for %s: %s", url, e)
    return ""


def fetch_article_text(url: str, timeout: int = 8) -> str:
    """Public wrapper to fetch article text. Keeps internal implementation separate."""
    return _fetch_article_text(url, timeout=timeout)


def collect_news_from_naver_search(query: str = "카드론 대출", limit: int = 40) -> list[dict[str, str]]:
    search_url = (
        "https://search.naver.com/search.naver?where=news&query="
        + requests.utils.quote(query)
    )
    try:
        resp = requests.get(search_url, timeout=10, headers=_BROWSER_HEADERS)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    if BeautifulSoup is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def append_item(anchor) -> None:
        href = str(anchor.get("href", "")).strip()
        if not href or href in seen:
            return
        if "channelPromotion" in href or href.startswith("javascript:"):
            return

        is_candidate = (
            "n.news.naver.com" in href
            or "news.naver.com" in href
            or "ref=naver" in href
            or "v.daum.net/v/" in href
        )
        if not is_candidate:
            return

        title = (anchor.get("title") or anchor.get_text(" ", strip=True) or "").strip()
        title = _clean_text(title)
        if not title or len(title) < 8:
            return
        if title in {"네이버뉴스", "네이버 뉴스", "뉴스", "기사원문", "기사 원문"}:
            return

        summary = ""
        container = anchor.find_parent(["div", "li", "article"])
        if container is not None:
            for selector in (
                ".news_dsc",
                ".dsc_wrap",
                ".api_txt_lines.dsc_txt_wrap",
                ".news_contents .dsc_txt_wrap",
                ".news_area .api_txt_lines",
            ):
                node = container.select_one(selector)
                if node is None:
                    continue
                candidate = _clean_text(node.get_text(" ", strip=True))
                if candidate and candidate != title:
                    summary = candidate
                    break

        seen.add(href)
        items.append(
            {
                "title": title,
                "summary": summary,
                "link": href,
                "published": "",
                "content": "",
                "query": query,
                "collector": "naver_search",
            }
        )

    headline_anchors = soup.select("a.news_tit")
    if headline_anchors:
        for anchor in headline_anchors:
            append_item(anchor)
            if len(items) >= limit:
                break
        return items

    for anchor in soup.find_all("a", href=True):
        append_item(anchor)
        if len(items) >= limit:
            break

    return items


def _normalize_news_key(item: dict[str, str]) -> tuple[str, str]:
    title = _clean_text(item.get("title", "")).lower()
    link = str(item.get("link", "")).strip().lower()
    return title, link


def collect_news(
    queries: list[str] | None = None,
    per_query_limit: int = 12,
    max_items: int = 80,
):
    """Collect news candidates using direct article links first.

    Primary source: Naver news search HTML (gives article URLs directly).
    Google News RSS is intentionally excluded because those links do not
    reliably yield article bodies during downstream crawling.
    Actual article content is fetched separately in the background.
    """
    effective_queries = [q for q in (queries or DEFAULT_NEWS_SEARCH_QUERIES) if str(q).strip()]
    collected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for query in effective_queries:
        query_items = collect_news_from_naver_search(query=query, limit=per_query_limit)
        for item in query_items:
            key = _normalize_news_key(item)
            if key in seen:
                continue
            seen.add(key)
            collected.append(item)
            if len(collected) >= max_items:
                return collected

    return collected


def analyze_news(news):
    keywords = ["연체", "금리", "규제", "DSR", "카드론", "신용대출", "사업자대출", "대출", "저축은행"]

    issues = []

    for n in news:
        for k in keywords:
            if k in n.get("title", ""):
                issues.append(n.get("title", ""))

    return issues
