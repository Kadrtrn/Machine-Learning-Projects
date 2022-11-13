import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Veri setinde değişken isimleri olmadığı için header=None verildi
sonar_data=pd.read_csv('sonar-data.csv',header=None)

sonar_data.head()

print(sonar_data.shape)
print(sonar_data.describe())

print(sonar_data[60].value_counts())

sonar_data.groupby(60).mean()

X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

print(X.shape,X_train.shape ,X_test.shape)


#Model Traning ---> Logistik Regression

model=LogisticRegression()

model.fit(X_train , Y_train)

y_pred=model.predict(X_test)

print(accuracy_score(Y_test, y_pred))

print(y_pred[:10])








#Making A Predictive System

input_data=(0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

#changing the input_data to a numpy array

input_data_as_np=np.asarray(input_data)

#reshape the np array as we are predicting for one instance

input_data_reshaped=input_data_as_np.reshape(1,-1)



prediction=model.predict(input_data_reshaped)

print(prediction)






"""When we have more data"""


input_data=([0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062],[0.0270,0.0163,0.0341,0.0247,0.0822,0.1256,0.1323,0.1584,0.2017,0.2122,0.2210,0.2399,0.2964,0.4061,0.5095,0.5512,0.6613,0.6804,0.6520,0.6788,0.7811,0.8369,0.8969,0.9856,1.0000,0.9395,0.8917,0.8105,0.6828,0.5572,0.4301,0.3339,0.2035,0.0798,0.0809,0.1525,0.2626,0.2456,0.1980,0.2412,0.2409,0.1901,0.2077,0.1767,0.1119,0.0779,0.1344,0.0960,0.0598,0.0330,0.0197,0.0189,0.0204,0.0085,0.0043,0.0092,0.0138,0.0094,0.0105,0.0093])

#changing the input_data to a numpy array

input_data_as_np=np.asarray(input_data)

#reshape the np array as we are predicting for one instance

input_data_reshaped=input_data_as_np.reshape(2,-2)



prediction=model.predict(input_data_reshaped)

print(prediction)












