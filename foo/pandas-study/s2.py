import pandas as pd
import  datetime
if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    home_data = pd.read_csv('test.csv')
    print(datetime.datetime.now().year - home_data.YrSold.max())
