import os
import glob

if __name__ == '__main__':
    print(os.path.expanduser('~'))  # s.path.expanduser() 函数将会使用 ~ 代表当前用户的主目录【Home Directory】
    print(os.path.join(os.path.expanduser('~'), 'humansize.py'))
    print(os.path.join(os.path.expanduser('~'), 'diveintopython3', 'examples', 'humansize.py'))
    (dirname, filename) = os.path.split(os.getcwd())
    print((dirname, filename))
    print("----------")
    (shortname, extension) = os.path.splitext(filename)
    print((shortname, extension))
    os.chdir(dirname)
    print(glob.glob('*/*.py'))
    metadata = os.stat(os.getcwd())
    print(os.getcwd())
    print(metadata)
    print(os.path.realpath('derivation.py'))
