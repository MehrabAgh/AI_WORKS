import numpy as np
import pandas as pd
import datetime as dt

seri = pd.Series([1,2 ,3 ,4 , 5])
print(seri,'\n')


df_arr = pd.DataFrame(np.array([["t1" , "t2" , "t3"],[1,2,3]]))
print(df_arr,'\n')


timeNow = dt.datetime.now()
firstDict = dict({"id" : 0 , "name" : "mehrab" , "year" : timeNow.now()})
df_1 = pd.DataFrame(data=firstDict , index=[0]) 
'''index for counter'''
print(df_1,'\n')


data_1 = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=[("a", "float"), ("b", "int"), ("c", "byte")])
df_2 = pd.DataFrame(data_1 , columns=["a","b" , "c"])
print(df_2,'\n')


data_2 = [{"id":0 , "name":"ai" ,"price" : 2500},
          {"id":1 , "name":"gamedev" ,"price" : 700},
          {"id":2 , "name":"python" ,"price" : 1400},
          {"id":3 , "name":"iot" ,"price" : 8000},
          {"id":4 , "name":"network" ,"price" : 4500},
          {"id":5 , "name":"security" ,"price" : 6000},
          {"id":6 , "name":"os" ,"price" : None},
          {"id":7 , "name":"database" ,"price" : 5500}]

df_3 = pd.DataFrame(data=data_2)

# print(df_3.head(2))
# print(df_3.tail(2),'\n')

# print(df_3.index)
# print(df_3.columns,'\n')

# print(df_3.to_numpy())
# print(df_3.describe(),'\n')

# print(df_3.T ,'\n')

# print(df_3.sort_index(),'\n')
# print(df_3.sort_values(by="name"),'\n')

# # select by label
# print(df_3["name"])
# print(df_3.loc[0:3 , ["name" , 'price']], '\n')
# """print(df_3.iat[6])"""

# # select by position
# print(df_3.iloc[6,0])
# print(df_3.iat[6,0])

# #select by condition
# print(df_3[df_3['price']>2500] , '\n')
# print(df_3[df_3['price'].isin([6000 , 5500])])

# #setting value
# df_3.loc[df_3['price'] == 5500 , "name"] = "blockchain"
# print(df_3 , '\n')

# #ploting
# #df_2.plot()

# #missing data
# print(df_3.dropna(how="any") , '\n')
# print(df_3.fillna(value=10000))

# #operation & function
# print(df_3.value_counts('price'), '\n')
# print(df_3['name'].str.upper(),'\n')

#grouping

#reshaping

df_3 = df_3.fillna(15000)
base_data = df_3.astype({"price" : "int"}) 

newData = df_3

DataConcated = pd.concat([newData , df_1])

DataConcated = DataConcated.fillna({'year' : 0 , 'price' : 0}) 
print(DataConcated)

res = DataConcated.to_excel("../datas/myData.xlsx" , sheet_name = "firstSheet")
print(res)
req = pd.read_excel("../datas/myData.xlsx" , index_col= None , na_values=['NA'])
print(req)

'''calculate'''
allSum = np.sum(newData['price'])
print(int(allSum))



#categoricals

#import & export data
# df_3.to_excel("foo.xlsx", sheet_name="Sheet1")
# pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])

# pd.read_csv("foo.csv")
# df_3.to_csv("foo.csv")















