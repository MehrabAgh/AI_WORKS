from MyModule import myOBJ
import pandas as pd

dataDoll = pd.read_csv('../datas/dollar_price.csv')

df = pd.DataFrame(dataDoll)
print(df)
df = df.fillna(0)
prices = df["Low"]
print(prices)