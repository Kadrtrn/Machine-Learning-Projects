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
import xgboost
from xgboost import XGBRegressor

xgb=XGBRegressor().fit(X_train,y_train)
xgb

y_pred=xgb.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))
print('İlkel hata:',error)

#Model Tuning

#learning rate overfitting i engellemek adına kulllanılır daraltma adım boyunu ifade eder 
#colsample_bytree oluşturulacak olan ağaçlarda değişkenlerden alınacak olan alt küme oranını ifade ediyor 

xgb_params={'learning_rate':[0.2,0.01,0.5],
            'max_depth':[2,3,4,5,8],
            'n_estimators':[100,200,500],
            'colsample_bytree':[0.4,0.7,1]
            
            }

xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1).fit(X_train,y_train)
print(xgb_cv_model.best_params_)
"""
Ağaç sayısının daha yüksek sayılarda olması beklenirdi 
fakat elimizdeki veri setlerinin boyutları ve yapısı elimizde ki algoritmaların yapısıyla bazne örtüşmeyebilir 
bazen çok kuvvetli bir algoritma çok basit kolay bir veri seti üzerinde kullanıllmaya çalışıldığında algoritma 
kendi kompleks problemleri uyarlanmak içn olulturulan yapısı itibariyle  bu tür problemlerde çok iyi sonuç gösteremeyebilir 
tam tersi KNN gibi görece biraz daha basit problemlerde iyi çalışan bir algoritma çok kompleks veri setleri için iyi çalışmayabilir
dolayısıyla burda 100 tahminci ağaç sayısı  çok beklenmedik olabilir 
"""
xgb_tuned=XGBRegressor(learning_rate=xgb_cv_model.best_params_['learning_rate'],
                       max_depth=xgb_cv_model.best_params_['max_depth'],
                       n_estimators=xgb_cv_model.best_params_['n_estimators'],
                       colsample_bytree=xgb_cv_model.best_params_['colsample_bytree']
                       ).fit(X_train,y_train)

y_pred=xgb_tuned.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))

print('Yeni test hatası : ',error)



































