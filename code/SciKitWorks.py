from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model  import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np


# add data for input
data_doller = pd.read_excel("../datas/dollar_price.xlsx")

# Removing rows with more than 1 null
df = data_doller[data_doller.isna().any(axis=1)] 

#print(data_doller, df)

# data templating(convert properties to array)
openprice_data = np.array(data_doller['Reopening'])
finalprice_data = np.array(data_doller['Final'])

# data templating(convert column to data frame)
df_openprice_data = pd.DataFrame(openprice_data)


"""[fit : learn and check , transform : clean]"""

# data cleaning(change value nulls in dataframe)
SI = SimpleImputer(strategy='median')
newModelOP = SI.fit_transform(df_openprice_data)

# data cleaning(change value categorical type in dataframe)
OE = OrdinalEncoder(categories='auto')
categoryConvert = OE.fit_transform(data_doller[['state']])

OH = OneHotEncoder()
bb = OH.fit_transform(data_doller[['state']])

LE = LabelEncoder()
finalprice_data = LE.fit_transform(y=finalprice_data)
print(finalprice_data)
finalprice_data_dec = LE.inverse_transform(y=finalprice_data)
print(finalprice_data_dec)
# print(data_doller['state'].unique())
