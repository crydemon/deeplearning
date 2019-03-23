def a():
    param = 'b'  # 这里就会出现这样的提示，因为在main定义的param对象被重新指定了新的值
    print(param)


if __name__ == '__main__':
    param = 'a'
    a()
