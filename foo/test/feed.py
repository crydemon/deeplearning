import feedparser

if __name__ == '__main__':
    d = feedparser.parse('https://newyork.craigslist.org')
    print(d['feed'])
