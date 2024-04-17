# 1 - import packages : 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
# 2 - reed the file :

train = pd.read_csv(r"C:\Users\moham\Desktop\MY AI\ML Projects\Airline Passenger Satisfaction\train.csv")
test = pd.read_csv(r"C:\Users\moham\Desktop\MY AI\ML Projects\Airline Passenger Satisfaction\test.csv")
df = pd.concat([train, test])
# 3 - get some info about the df :

# print(df.shape)
# print(df.info())
# print(df.sample(5))
# print(df.describe())
# print(len(df['Gender'].unique()))
# print(len(df['Customer Type'].unique()))
# print(len(df['Type of Travel'].unique()))
# print(len(df['Class'].unique()))
# print(len(df['satisfaction'].unique()))

# 4 - df Preprocessing :

df = df.drop(df.columns[0:2], axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])
df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
df['Class'] = label_encoder.fit_transform(df['Class'])
df['satisfaction'] = label_encoder.fit_transform(df['satisfaction'])

# 5 - split the df :

X = df.drop(['satisfaction'], axis=1)
Y = df['satisfaction']

# 6 - train the df :

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20 , random_state=42)

# 7 - use Decision Tree model  to predict :

from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(x_train,y_train)
y_pred = DT_model.predict(x_test)

# 8 - evaluate the model :

from sklearn.metrics import confusion_matrix
from sklearn import metrics

confusion_matrix1 = confusion_matrix(y_test, y_pred)
# print(confusion_matrix1)

accuracy  = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

# 9 - save the model :

import pickle
pickle.dump(DT_model,open('Passenger_Satisfaction.pkl','wb'))
model = pickle.load(open( 'Passenger_Satisfaction.pkl', 'rb' ))