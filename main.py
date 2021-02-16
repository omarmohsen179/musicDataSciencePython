import pandas as pd
from sklearn.tree import DecisionTreeClassifier
music_data= pd.read_csv('music.csv')

x= music_data.drop(columns=['genre'])
y=music_data['genre']
model=DecisionTreeClassifier()
model.fit(x,y)
pred=model.predict([[21,1],[22,0]])
print(pred)
