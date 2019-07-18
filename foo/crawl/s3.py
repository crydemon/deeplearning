import requests
import time
import random
import re

# 请求头
headers = [{
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'},
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'},
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'},
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'}
]


# 获取网页源码
def get_text(url):
    try:
        response = requests.get(url, headers=random.choice(headers))
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except:
        return ''


# 爬取评论
def crawl_comments(urls, n):
    print('现在正在爬 {} 星评论：'.format(str(n)))
    page = 1
    for url in urls:
        txt = get_text(url)
        if txt == '':
            print('第 {} 页请求失败！'.format(str(page)))
            page += 1
            continue
        print('正在爬取第 {} 页---------------------'.format(str(page)))

        # 获取这一页所有评论用户的 id
        ids = re.findall('<div class="main review-item" id="(.*?)">', txt)

        # 访问评论的每一个网页，并爬取信息
        full_comment_url = 'https://movie.douban.com/j/review/{}/full'
        count = 1  # 记录爬取到了当前这一页的第几条评论
        for id in ids:
            full_url = full_comment_url.format(id)
            html = get_text(full_url)
            if html == '':
                continue

            content_pattern = re.compile('data-original(.*?)main-author', re.S)
            content = re.findall(content_pattern, html)
            text_pattern = re.compile('[\u4e00-\u9fa5|。]+', re.S)
            text = re.findall(text_pattern, content[0])
            text = ''.join(text)
            text = text.replace('所有任何形式转载请联系作者', '').replace('本文版权归作者', '')

            print('正在写入第 {} 条'.format(str(count)))
            count += 1

            # 写入文件
            write_txt(text)

        page += 1
        time.sleep(random.randint(1, 3))


# 写入 txt 文本
def write_txt(comment):
    # 将文本分成多行写入
    comment = comment.split('。')
    with open('liulangdiqiu.txt', 'w', encoding='utf-8') as f:
        for c in comment:
            print(c)
            f.write(c + '\n')


# 主函数
def main():
    # 1-5 星的 urls 的构造
    urls_star_1 = ['https://movie.douban.com/subject/26266893/reviews?rating=1&start={}'.format(str(page)) \
                   for page in range(0, 981, 20)]
    urls_star_2 = ['https://movie.douban.com/subject/26266893/reviews?rating=2&start={}'.format(str(page)) \
                   for page in range(0, 761, 20)]
    urls_star_3 = ['https://movie.douban.com/subject/26266893/reviews?rating=3&start={}'.format(str(page)) \
                   for page in range(0, 2081, 20)]
    urls_star_4 = ['https://movie.douban.com/subject/26266893/reviews?rating=4&start={}'.format(str(page)) \
                   for page in range(0, 3961, 20)]
    urls_star_5 = ['https://movie.douban.com/subject/26266893/reviews?rating=3&start={}'.format(str(page)) \
                   for page in range(0, 9001, 20)]

    # 爬取一星评价
    # crawl_comments(urls_star_1,n = 1)

    # 爬取二星评价
    # crawl_comments(urls_star_1, n=2)

    # 爬取三星评价
    # crawl_comments(urls_star_1, n=3)

    # 抓取四星评价
    # crawl_comments(urls_star_1, n=4)

    # 抓取五星评价
    crawl_comments(urls_star_1, n=5)


# 主接口
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
