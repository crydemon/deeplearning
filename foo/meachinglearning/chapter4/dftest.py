import pandas as pd

if __name__ == '__main__':
    df = pd.DataFrame([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], columns=["col1", "col2", "col3", "col4"])
    print(df)
    print("-------------")
    print(df.mean(axis=0))  # 列
    print("-------------")
    print(df.mean(axis=1))  # 行
    print("-------------")
    print(df.drop(0))  # 删除行，
    print("-------------")
    print(df.drop(['col1'], axis=1))  # 删除列，
