# !/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import datetime


def from_mysql_get_all_info():
    from pyhive import hive
    conn = hive.Connection(host='internal-vova-bd-spark-ts-438167675.us-east-1.elb.amazonaws.com', port=10001,
                           username='hadoop', database='tmp')
    cursor = conn.cursor()
    cursor.execute('select * from tmp.tmp_goods_select_a_final limit 10')
    data = cursor.fetchall()
    cols = cursor.description
    # 执行
    conn.commit()
    conn.close()
    # 将数据truple转换为DataFrame
    col = []
    for i in cols:
        col.append(i[0])
    data = [tuple(col)] + data
    return data


def write_csv():
    import csv
    data = from_mysql_get_all_info()
    filename = 'goods_a.csv'
    with open(filename, mode='w') as f:
        write = csv.writer(f, dialect='excel')
        for item in data:
            write.writerow(item)

write_csv()
