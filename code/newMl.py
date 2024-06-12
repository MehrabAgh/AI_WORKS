import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_game = pd.read_csv("../data/ratings.csv")

X = data_game.drop(columns='Developer')
y = data_game['Developer']

x_train , x_test , y_train , y_test = train_test_split(X,y, test_size=.2)

model = DecisionTreeClassifier()

model.fit(x_train,y_train)

pred = model.predict(x_test)

print(pred)

