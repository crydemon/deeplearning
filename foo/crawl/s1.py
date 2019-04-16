import requests
import pdfkit
import os
from bs4 import BeautifulSoup

if __name__ == '__main__':
    target = 'http://www.cnblogs.com/zangrunqiang/p/5515872.html'
    req = requests.get(url=target)
    html = req.text
    bf = BeautifulSoup(html)
    content = bf.find_all('blogpost-body')[0]
    # 渲染的html模板
    html_template = """
       <!DOCTYPE html>
       <html lang="en">
       <head>
           <meta charset="UTF-8">
       </head>
       <body>
       {content}
       </body>
       </html>
       """
    html = html_template.format(content=content)
    html = html.encode("UTF-8")
    out_file = r'f:\out.html'
    fw = open(out_file, 'wb')
    fw.write(html)
    fw.close()

    config = pdfkit.configuration(wkhtmltopdf=r'F:\apps\wkhtmltopdf\bin\wkhtmltopdf.exe')
    pdfkit.from_url(out_file, r'f:\out.pdf', configuration=config)
    os.remove(out_file)
    print("done")
