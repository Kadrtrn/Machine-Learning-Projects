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


#Tek değişken seçip CART algoritmasını anlamaya çalışalım 

#***************************************************
X_train=pd.DataFrame(X_train['Hits'])
X_test=pd.DataFrame(X_test['Hits'])

#max_leaf_nodes dallanma sayısını ifade ediyor -----ve grafikteki dallanma azalıyor 
#Bölünmeler arttıkça tahmin kuvettelenecektir 
cart_model=DecisionTreeRegressor(max_leaf_nodes=3)
cart_model.fit(X_train,y_train)


#Aşagıdaki görselleştirme kodları konu kapsamında değil !!
X_grid=np.arange(min(np.array(X_train)),max(np.array(X_train)),0.01)
X_grid=X_grid.reshape((len(X_grid),1))

plt.scatter(X_train,y_train,color='red')

plt.plot(X_grid,cart_model.predict(X_grid),color='blue')

plt.title('CART REGRESYON AĞACI')
plt.xlabel('Atış Sayısı (Hits)')
plt.ylabel('Maaş (Salary)')

#Grafiği tam anlamıyorsan videoya bak tekrar !!!!!!
#Tek değişken için tahmin

y_pred=cart_model.predict(X_test)

err=np.sqrt(mean_squared_error(y_test, y_pred))
print('Tek değişken için hata (parametre verilmedi)',err)
#********************************


#Tüm değişkenler için CART model


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

#DecisionTreeRegressor e random state verilince sonuçlar farklı çıkmıyor
cart_model=DecisionTreeRegressor().fit(X_train,y_train)
y_pred=cart_model.predict(X_test)
error=np.sqrt(mean_squared_error(y_test, y_pred))

print('Tüm değişkenller için hata (Parametre verilmedi)',error)




#Model tuning

cart_params={'max_depth':[2,3,4,6,7],
             'min_samples_split':[2,10,5,30,50,10,60,51,52,53,54]}

cart_model=DecisionTreeRegressor()
cart_cv_model=GridSearchCV(cart_model, cart_params,cv=10).fit(X_train,y_train)
print(cart_cv_model.best_params_)

#final model
cart_model=DecisionTreeRegressor(max_depth = cart_cv_model.best_params_['max_depth'],
                                 min_samples_split=cart_cv_model.best_params_['min_samples_split']).fit(X_train,y_train)

y_pred=cart_model.predict(X_test)
error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error) 
    





































