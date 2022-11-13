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



gbm_model=GradientBoostingRegressor().fit(X_train,y_train)

gbm_model


y_pred=gbm_model.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))
print('İlkel hata değeri',error)


#Model Tuning

#criterion bölünmelerle ilgili saflık ölççüsünü ifade etmekte 
#learning_rate ağaçların katkısıyla ilgili
#loss = kayıp fonksiyonu ls en küçük kareleri ifade etmekte # lad huber quantile dayanıklı yöntemlerdir yani aykırı gözlemlere karşı dayanıklı
#supsample oluşturulacak olan ağaçları oluşştururken göz önünde bulunacak olan oran 1 yazdığımızda hepsini dahil ederek ağaç oluştumuş oluyor

gbm_params={'learning_rate':[0.001,0.1,0.01],
            'n_estimators':[100,200,230],
            'subsample':[1,0.5,0.8],
            'loss':['ls','lad','quantile']
            }


gbm_cv_model=GridSearchCV(gbm_model,gbm_params,cv=10,n_jobs=-1).fit(X_train, y_train)

print(gbm_cv_model.best_params_)


gbm_tuned=GradientBoostingRegressor(learning_rate=gbm_cv_model.best_params_['learning_rate'],
                                    n_estimators=gbm_cv_model.best_params_['n_estimators'],
                                    subsample=gbm_cv_model.best_params_['subsample'],
                                    loss=gbm_cv_model.best_params_['loss']
                                    ).fit(X_train,y_train)


y_pred=gbm_tuned.predict(X_test)
error=np.sqrt(mean_squared_error(y_test, y_pred))

print('Yeni hata değerri :',error)


#Değişkenlerin Önem Düzeyleri


Importance=pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100},
                        index=X_train.columns
                        )

Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',
                                           color='r')
            
plt.xlabel('Variable Importance')
plt.gca().legend_=None

































