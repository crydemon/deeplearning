import pandas as pd

if __name__ == '__main__':
    f_path = 'd:/datanb.csv'
    df = pd.read_csv(f_path, header=0, sep=',')


    print(df['up_login'][1])
