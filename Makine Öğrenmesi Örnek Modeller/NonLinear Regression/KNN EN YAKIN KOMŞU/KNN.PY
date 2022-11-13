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



knn_model=KNeighborsRegressor().fit(X_train,y_train)

#Komşu sayısı vs hepsini kendi verdi yukarıda 
#Komşu sayısı görme
print(knn_model.n_neighbors)
#Kullanılmış olan metrigi görme.
print(knn_model.metric)


#Tahmin etme 
y_pred=knn_model.predict(X_test)

print(y_pred[0:5])

#hATA BULMA 
#hata kareler ortalamasının karekökü
error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)


#Model Tuning

#EL yordamıyla yapıyoruz suan 
RMSE=[]

for k in range(10):
    k = k+1
    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    y_pred=knn_model.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE.append(rmse)
    print('k=',k,'için RMSE değeri:',rmse)

#El yordamı bitti

"""
*****      GridSearchCV     ***********
Kullanacak oldupumuz makine öğrenmesi 
algoritmalarında belirleme çalıştığımız
 hiperparmetrelerin belirlenmesi için kullanılan fonksiyondur
Birbirinden farklı ve fazla sayıda parametre olduğunda onları bir grid mantığıyla 
ızgara mantığıgla değerlendirip olası tüm parametre kombinasyonları 
değerlendirmek ve kıyaslama yapmak için kullanılan bir fonksiyondur
"""
knn_params={"n_neighbors":np.arange(1,30,1)}

knn=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn, knn_params,cv=10).fit(X_train,y_train)

print(knn_cv_model.best_params_)

#En iyi parameyi bulduğumuza göre en son modeli kuralım 
#parametreyi yazmaya dikkat et içinde başka parametrelerde olabilir diye öyle yazıldı
knn_tuned=  KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_['n_neighbors']).fit(X_train,y_train)

y_pred=knn_tuned.predict(X_test)
err=np.sqrt(mean_squared_error(y_test, y_pred))
print(err)































































