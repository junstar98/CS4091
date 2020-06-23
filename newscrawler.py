from korea_news_crawler.articlecrawler import ArticleCrawler

Crawler = ArticleCrawler()
Crawler.set_category("IT과학")
Crawler.set_date_range(2012, 1, 2012, 12)
Crawler.start()
