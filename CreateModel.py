


#train_test_splite randomly pick random test so value is changeable
#testsize 0.2 we only test 20 percent of our data and train 80 percent
# the more we train our data we get more accuracy but more performance



import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

#import the data
def createModel():
    music_data = pd.read_csv('music.csv')
#clean data to features x and label y
    x = music_data.drop(columns=['genre'])
    y = music_data['genre']
#splite data to train and test much we test much less accuracy
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#create model
    model = DecisionTreeClassifier()
#train model
    model.fit(X_train, y_train)
#create dot file to understand how model workes
    tree.export_graphviz(model,out_file='music-recommender.dot'
                         ,feature_names=['age','gender']
                         ,label='all'
                         ,class_names=sorted(y.unique())
                         ,rounded=True,
                         filled=True)
#make prediction
    prediction = model.predict(X_test)
#y_test = expected values
#prediction= acuale model values
    joblib.dump(model, 'music-recommender.joblib')
    accuracy = accuracy_score(y_test, prediction)
    print("model accuracy : ",accuracy)