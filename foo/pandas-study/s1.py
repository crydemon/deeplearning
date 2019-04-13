import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 全部输出
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    s = pd.Series([1, 2, 3, np.nan, 5, 6])
    print(s)  # 索引在左边 值在右边
    dates = pd.date_range('20180310', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])  # 生成6行4列位置
    print(df)  # 输出6行4列的表格
    print(df['B'])
    df_1 = pd.DataFrame({'A': 1.,
                         'B': pd.Timestamp('20180310'),
                         'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                         'D': np.array([3] * 4, dtype='int32'),
                         'E': pd.Categorical(["test", "train", "test", "train"]),
                         'F': 'foo'
                         })
    print(df_1)
    print(df_1.dtypes)
    print(df_1.index)  # 行的序号
    # Int64Index([0, 1, 2, 3], dtype='int64')
    print(df_1.columns)  # 列的序号名字
    # Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
    print(df_1.values)  # 把每个值进行打印出来
    print(df_1.describe())  # 数字总结

    # dau = pd.read_csv('d:/dau.csv')
    # new_user_dau = pd.read_csv('d:/new_user_dau.csv')
    # # pd inner join
    # tmp = pd.merge(dau, new_user_dau, on=['action_date'])
    # tmp['new_user_rate'] = tmp['total_nums_y'] / tmp['total_nums_x']
    # tmp['old_user_rate'] = 1 - tmp['new_user_rate']
    # tmp = tmp.rename(columns={'total_nums_x': 'total_nums', 'total_nums_y': 'new_user_nums'})
    # print(tmp)

    print(df.loc['20180312', ['A', 'B']])  # 按照行标签进行选择 精确选择
    print(df.iloc[3, 1])  # 输出第三行第一列的数据
    print(df.iloc[3:5, 0:2])  # 进行切片选择
    print(df.iloc[[1, 2, 4], [0, 2]])  # 进行不连续筛选
    #  print(df.ix[:3, ['A', 'C']]) 已经丢弃
    print(df[df.A > 0])  # 筛选出df.A大于0的元素 布尔条件筛选

    dates = pd.date_range('20180310', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    print(df)

    df.iloc[2, 2] = 999  # 单点设置
    df.loc['2018-03-13', 'D'] = 999
    print(df)
    df[df.A > 0] = 999  # 将df.A大于0的值改变
    print(df)
    df['F'] = np.nan
    print(df)
    df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20180310', periods=6))  # 增加一列
    print(df)
    dates = pd.date_range('20180310', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    df.iloc[0, 1] = np.nan
    df.iloc[1, 2] = np.nan
    print(df)
    print(df.dropna(axis=0, how='any'))  # 0对行进行操作 1对列进行操作 any:只要存在NaN即可drop掉 all:必须全部是NaN才可drop

    print(df.fillna(value=0))  # 将NaN值替换为0
    print(pd.isnull(df))  # 矩阵用布尔来进行表示 是nan为ture 不是nan为false
    print(np.any(df.isnull()))  # 判断数据中是否会存在NaN值

    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
    res = pd.concat([df1, df2, df3], axis=0,
                    ignore_index=True)  # 0表示竖项合并 1表示横项合并 ingnore_index重置序列index index变为0 1 2 3 4 5 6 7 8
    print(res)
    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
    print(df1)
    print(df2)
    res = pd.concat([df1, df2], axis=1, join='outer')  # 行往外进行合并
    print(res)

    res = pd.concat([df1, df2], axis=1, join='outer')  # 行相同的进行合并
    print(res)

    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

    res = df1.append(df2, ignore_index=True)  # 将df2合并到df1的下面 并重置index
    print(res)
    res = df1.append(s1, ignore_index=True)  # 将s1合并到df1下面 并重置index
    print(res)
