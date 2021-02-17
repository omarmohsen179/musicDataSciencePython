

import joblib
import CreateModel
#create the model to privent the time of train and test every time run program

input=input("Enter 1 to test\nEnter 1 to run module\nExit zero\n")

if int(input)==1:
    model=joblib.load('music-recommender.joblib')
    predictionEx=model.predict([[21,1],[22,0]])
    print( predictionEx)

else:
    CreateModel.createModel()


