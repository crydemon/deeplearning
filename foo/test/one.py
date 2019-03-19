# 1. 如果模块是被导入，__name__的值为模块名字
# 2. 如果模块是被直接执行，__name__的值为’__main__’
# 在Python中，一个.py文件就是一个模块，一般情况下，模块的名字就是文件名(不包括扩展名.py)。
# 全局变量__name__存放的就是模块的名字。而特殊情况就是，
# 当一个模块作为脚本执行时或者在交互式环境中，
# 如Ipython、Python自带的shell等直接运行代码，__name__的值不再是模块名，
# 而是__main__。__main__是顶层代码执行作用域的名字。

def func():
    print("func() in one.py")


print("top-level in one.py")

if __name__ == "__main__":
    print("one.py is being run directly")
else:
    print(__name__)
    print("one.py is being imported into another module")
