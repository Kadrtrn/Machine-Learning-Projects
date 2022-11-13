import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.metrics import accuracy_score , classification_report
from sklearn.neighbors import KNeighborsClassifier


import warnings

warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

df=pd.read_csv('./diabetes.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


y=df['Outcome']
X=df.drop(['Outcome'],axis=1)

X_train ,X_test ,y_train,y_test=train_test_split(
                                                    X,y,
                                                    test_size=0.3,
                                                    random_state=42)

print(df.head())


#Model ve Tahmin
knn_model=KNeighborsClassifier().fit(X_train,y_train)

y_pred=knn_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

#print(classification_report(y_test, y_pred))


#Model Tuning

knn_params={'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()
knn_cv_model=GridSearchCV(knn,knn_params,cv=10).fit(X_train, y_train)

print(knn_cv_model.best_score_)

print(knn_cv_model.best_params_)

knn_tuned=KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)

y_pred=knn_tuned.predict(X_test)
print(accuracy_score(y_test, y_pred))



#accuracy_score hesaplamak için farklı bir yaklaşım 

print(knn_tuned.score(X_test,y_test))



"""
Pregnancies=5
Glucose=89
BloodPressure= 69
SkinThickness=26
Insulin=90
BMI=30
DiabetesPedigreeFunction=0.617
Age=39
******
Değerleri bu şekilde olan birinin hasta olup olmadığını knn_tuned modelini kullanarak nasıl tahmin edebilirim
"""
print('*****')
data = [5,89,69,26,90,30.0,0.617,39] 
data = np.array(data).reshape(1,-1)
print(data)
print(knn_tuned.predict(data))




