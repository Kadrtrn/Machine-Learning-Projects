import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error
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

from lightgbm import LGBMRegressor


#Model ve Tahmin

lgbm=LGBMRegressor().fit(X_train,y_train)

lgbm

y_pred=lgbm.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))

print('İlkel test hatası: ',error)

print(lgbm) #yanına nokta koyup varsayılan parametre değerleri öğrenilebilir



#Model Tuning



lgbm_params={'learning_rate':[0.01,0.089,0.1,0.5,1],
            'max_depth':[1,2,3,4,5,6,7],
            'n_estimators':[20,40,100]            
            }


lgbm_cv_model=GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1).fit(X_train, y_train)

print(lgbm_cv_model.best_params_)


#Final Model
lgbm_tuned=LGBMRegressor(learning_rate=lgbm_cv_model.best_params_['learning_rate'],
                       max_depth=lgbm_cv_model.best_params_['max_depth'],
                       n_estimators=lgbm_cv_model.best_params_['n_estimators']
                       ).fit(X_train,y_train)


y_pred=lgbm_tuned.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))

print('Yeni test hatası:',error)

















































