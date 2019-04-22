# 在Python 3，所有的字符串都是使用Unicode编码的字符序列
import re

if __name__ == '__main__':
    s = '深入'
    print(len(s))
    s = '100 NORTH MAIN ROAD'
    s[:-4] + s[-4:].replace('ROAD', 'RD.')
    print(re.sub('ROAD$', 'RD.', s))
    pattern = '^M?M?M?(CM|CD|D?C?C?C?)$'
    print(re.search(pattern, 'MMMCCC'))
    # search()返回一个匹配对象。匹配对象中有很多的方法来描述这个匹配结果信息。如果没有匹配到，search()返回None。你只需要关注search()函数的返回值就可以知道是否匹配成功。
    phonePattern = re.compile(r'^(\d{3})-(\d{3})-(\d{4})$')
    print(phonePattern.search('800-555-1212').groups())
