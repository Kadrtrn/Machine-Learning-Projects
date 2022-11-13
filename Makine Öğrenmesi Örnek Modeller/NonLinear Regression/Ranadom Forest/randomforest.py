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



#Model Tahmin 

rf_model=RandomForestRegressor(random_state=42).fit(X_train,y_train)

rf_model

y_pred=rf_model.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)


#Model Tuning

#max_depth= amx deriblik max_features= bölünmelerde göz önünde bulundurulması gereken değişken sayısı
#n_estimators= ağaç sayısı 
#min_samples_split= 
rf_params={'max_depth':[5,8,10],
           'max_features':[2,5,10],
           'n_estimators':[200,300,400,500,100,200],
           'min_samples_split':[2,3,4,5,10,80,100]
           }

#max_features ve max_depth ile fazla oynama yapmak öönerimiyor



rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1).fit(X_train,y_train)

print(rf_cv_model.best_params_) 

rf_model=RandomForestRegressor(random_state=42,max_depth=rf_cv_model.best_params_['max_depth'],
                               max_features=rf_cv_model.best_params_['max_features'],
                               n_estimators=rf_cv_model.best_params_['n_estimators'],
                               min_samples_split=rf_cv_model.best_params_['min_samples_split'])

rf_tuned=rf_model.fit(X_train,y_train)
y_pred=rf_tuned.predict(X_test)
error=np.sqrt(mean_squared_error(y_test, y_pred))

print(error)


#Değişken Önem Düzeyi 
#Modelleme işlemleri sırasında göz önünde bulundurmamız yada odaklanmamız gereken değişkenleri görmek adına bir imkan sağlamaktadır 

Importance=pd.DataFrame({'Importance':rf_tuned.feature_importances_*100},
                        index=X_train.columns
                        )

Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',
                                           color='r')
            
plt.xlabel('Variable Importance')
plt.gca().legend_=None






















