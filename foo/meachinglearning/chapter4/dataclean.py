import pandas as pd

if __name__ == '__main__':
    f_path = 'd:/dataclean.csv'
    df = pd.read_csv(f_path, header=0, sep=',')
    for i in range(0, df.size-2):
        if df['up_login'][i] == 0:
            print(df['post_email'][i])
            print(i)
