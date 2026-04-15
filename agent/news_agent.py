import feedparser


def collect_news():

    url = "https://news.google.com/rss/search?q=카드론+대출&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)

    news = []

    for entry in feed.entries[:40]:
        news.append(
            {
                "title": entry.title,
                "summary": entry.summary,
                "link": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
            }
        )

    return news


def analyze_news(news):

    keywords = ["연체", "금리", "규제", "DSR", "카드론", "신용대출"]

    issues = []

    for n in news:
        for k in keywords:
            if k in n["title"]:
                issues.append(n["title"])

    return issues
