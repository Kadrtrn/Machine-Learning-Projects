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



from catboost import CatBoostRegressor

catb_model=CatBoostRegressor().fit(X_train,y_train)



y_pred=catb_model.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))

print('İlkel test hatası:',error)



#Model Tuning 

#iteration : kullanılan ağaç sayısı 
#depth derinlik
catb_params={'iterations':[200,500],
             'learning_rate':[0.01,0.1],
             'depth':[3,6]
             }

catb_cv_model=GridSearchCV(catb_model,catb_params,cv=5,n_jobs=-1).fit(X_train,y_train)

print(catb_cv_model.best_params_)
 


catb_tuned=CatBoostRegressor(depth=catb_cv_model.best_params_['depth'],
                             learning_rate=catb_cv_model.best_params_['learning_rate'],
                             iterations=catb_cv_model.best_params_['iterations'] 
                             ).fit(X_train,y_train)

y_pred= catb_tuned.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))

print('Yeni Test Hatası:',error)



















