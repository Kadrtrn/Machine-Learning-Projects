import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets

house_price_dataset=sklearn.datasets.load_boston()

df=pd.DataFrame(house_price_dataset.data,columns=house_price_dataset.feature_names)

df.head()


df.info()


df.describe()

# add the target (price) column to the DataFrame
df['price']=house_price_dataset.target

df.shape

df.isnull().sum()

X=df.drop(['price'],axis=1)
Y=df['price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)

xgb=XGBRegressor().fit(X_train,Y_train)


xgb_params={'learning_rate':[0.01,0.001],
            'max_depth':[7,8],
            'n_estimators':(1000,1300,1500),
            'colsample_bytree':[0.7,0.9,1.2]
            
            }


xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10).fit(X_train,Y_train)

xgb_cv_model.best_params_

xgb_tuned=XGBRegressor(learning_rate=xgb_cv_model.best_params_['learning_rate'],
                       max_depth=xgb_cv_model.best_params_['max_depth'],
                       n_estimators=xgb_cv_model.best_params_['n_estimators'],
                       colsample_bytree=xgb_cv_model.best_params_['colsample_bytree']
                       ).fit(X_train,Y_train)



y_pred=xgb_tuned.predict(X_test)
print('Mean Squared Error :',np.sqrt(mean_squared_error(Y_test,y_pred)))
print('R^2 Score :',r2_score(Y_test,y_pred))
print('Mean Absolute Error :',mean_absolute_error(Y_test, y_pred))




plt.scatter(Y_test,y_pred)
plt.xlabel('Real Data')
plt.ylabel('Predicted Data')
plt.show()




