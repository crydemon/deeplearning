# -*- coding:utf-8 -*-

import urllib.request
import http.cookiejar
import urllib.parse
import json
import time
import codecs
from Crypto.Cipher import AES
import base64
import os
import pymysql


class Music:

    # 初始化
    def __init__(self):
        # 设置代理，以防止本地IP被封
        self.proxyUrl = "http://181.176.161.19:8080"
        # request headers,这些信息可以在ntesdoor日志request header中找到，copy过来就行
        self.Headers = {'Cookie': 'appver=1.5.0.75771;', 'Referer': 'http://music.163.com/'}

        # 使用http.cookiejar.CookieJar()创建CookieJar对象
        self.cjar = http.cookiejar.CookieJar()
        # 使用HTTPCookieProcessor创建cookie处理器，并以其为参数构建opener对象
        self.cookie = urllib.request.HTTPCookieProcessor(self.cjar)
        self.opener = urllib.request.build_opener(self.cookie)
        # 将opener安装为全局
        urllib.request.install_opener(self.opener)
        # 第二个参数
        self.second_param = "010001"
        # 第三个参数
        self.third_param = "00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7"
        # 第四个参数
        self.forth_param = "0CoJUm6Qyw8W8jud"

        # 用户名：root 密码：123456 数据库名：aqi-changsha
        self.db = pymysql.connect("localhost", "root", "root", "jupiter", charset="utf8")
        self.cursor = self.db.cursor()

    def get_params(self, page):
        # 获取encText，也就是params
        iv = "0102030405060708"
        first_key = self.forth_param
        second_key = 'F' * 16
        if page == 0:
            first_param = '{rid:"", offset:"0", total:"true", limit:"20", csrf_token:""}'
        else:
            offset = str((page - 1) * 20)
            first_param = '{rid:"", offset:"%s", total:"%s", limit:"20", csrf_token:""}' % (offset, 'false')
        self.encText = self.AES_encrypt(first_param, first_key, iv)
        self.encText = self.AES_encrypt(self.encText.decode('utf-8'), second_key, iv)
        return self.encText

    def AES_encrypt(self, text, key, iv):
        # AES加密
        pad = 16 - len(text) % 16
        text = text + pad * chr(pad)
        encryptor = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
        encrypt_text = encryptor.encrypt(text.encode('utf-8'))
        encrypt_text = base64.b64encode(encrypt_text)
        return encrypt_text

    def get_encSecKey(self):
        # 获取encSecKey
        encSecKey = "257348aecb5e556c066de214e531faadd1c55d814f9be95fd06d6bff9f4c7a41f831f6394d5a3fd2e3881736d94a02ca919d952872e7d0a50ebfa1769a7a62d512f5f1ca21aec60bc3819a9c3ffca5eca9a0dba6d6f7249b06f5965ecfff3695b54e1c28f3f624750ed39e7de08fc8493242e26dbc4484a01c76f739e135637c"
        return encSecKey

    def get_json(self, url, params, encSecKey):
        # post所包含的参数
        self.post = {
            'params': params,
            'encSecKey': encSecKey,
        }
        # 对post编码转换
        self.postData = urllib.parse.urlencode(self.post).encode('utf8')
        try:
            # 发出一个请求
            self.request = urllib.request.Request(url, self.postData, self.Headers)
        except urllib.error.HTTPError as e:
            print(e.code)
            print(e.read().decode("utf8"))
        # 得到响应
        self.response = urllib.request.urlopen(self.request)
        # 需要将响应中的内容用read读取出来获得网页代码，网页编码为utf-8
        self.content = self.response.read().decode("utf8")
        # 返回获得的网页内容
        return self.content

    def close(self):
        self.db.close()

    def get_allcomments(self, url):
        # 获取全部评论
        params = self.get_params(1)
        encSecKey = self.get_encSecKey()
        content = self.get_json(url, params, encSecKey)
        json_dict = json.loads(content)
        print(json_dict)
        print("---------")
        comments_num = int(json_dict['total'])
        present_page = 0
        if comments_num % 20 == 0:
            page = comments_num / 20
        else:
            page = int(comments_num / 20) + 1
        print("共有%d页评论" % page)
        print("共有%d条评论" % comments_num)
        # 逐页抓取

        insert_data = "insert into comments(user_name, star, create_time, comment) values (%s,%s,%s,%s)"
        for i in range(int(page)):
            params = self.get_params(i + 1)
            encSecKey = self.get_encSecKey()
            json_text = self.get_json(url, params, encSecKey)
            json_dict = json.loads(json_text)
            present_page = present_page + 1

            datas = []
            for i in json_dict['comments']:
                # 将评论输出至txt文件中
                data = (i['user']['nickname'], int(i['likedCount']), int(i['time']), i['content'])
                datas.append(data)
            time.sleep(2)
            print("第%d页抓取完毕" % present_page)
            if datas is None:
                continue
            # print(datas)
            i = self.cursor.executemany(insert_data, datas)
            print(i)
            self.db.commit()


if __name__ == '__main__':
    music = Music()
    for x in range(227323, 309530090):
        music.get_allcomments("https://music.163.com/weapi/v1/resource/comments/R_SO_4_" + str(x) + "?csrf_token")
        time.sleep(1)
    music.close()
