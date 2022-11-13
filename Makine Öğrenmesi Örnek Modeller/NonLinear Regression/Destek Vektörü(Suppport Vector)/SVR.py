import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR


#Uyarı mesajlarını kaldırmak 
from warnings import filterwarnings
filterwarnings('ignore')


df=pd.read_csv('Hitters.csv')
df=df.dropna()

dms=pd.get_dummies(df[['League','Division','NewLeague']])

y=df['Salary']

X_=df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

X=pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,
                                               y,
                                               test_size=0.25,
                                               random_state=42)


#aşagıdaki linear bir svr modeli

svr_model=SVR(kernel='linear').fit(X_train, y_train)
print(svr_model)



svr_predict=svr_model.predict(X_train)

svr_predict=svr_model.predict(X_test)


#print(svr_predict)


print(svr_model.intercept_)
print(svr_model.coef_)


#test hatası 
y_pred=svr_model.predict(X_test)
error=np.sqrt(mean_squared_error(y_test,y_pred))
print(error)


#buradaki kernelimiz dogrusal istersek dogrusal olmayan rbf isimli 
#dogrusal olmayan radial basis function (isim yanlış olabilir ) adındaki değeride kullanabiliriz





######Model Tuning 

svr_model=SVR(kernel='linear')

#en iyi c yi bulalım
"""
svr_params={'C':[0.1,0.5,1,3]}

#verbose isimli arguman 2 değeri ile çalıştırırsak bu çalışma esnasındaki durumu raporlar
#njobs isimli bir arguman -1 değeri ile çalıştırılırsa bilgisayarımızda var olan işlemci gücü en yüksek performans ile çalışır
svr_cv_model=GridSearchCV(svr_model,svr_params,cv=5,verbose =2,n_jobs=-1).fit(X_train,y_train)
print(svr_cv_model.best_params_)

"""


#Final modelini oluşturalım 

svr_tuned=SVR(kernel='linear',C=0.5).fit(X_train,y_train)
y_pred=svr_tuned.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)
























