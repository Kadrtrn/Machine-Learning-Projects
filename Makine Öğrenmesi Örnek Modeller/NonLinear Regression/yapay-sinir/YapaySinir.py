import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


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


#Model & Tahmin
#Standartlaştırma yapacağız

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaled=scaler.transform(X_test)#Fit işlemi test setinden öncede yapılabilir dedi videoda fazla bir değişiklik olmuyor


#Hiçbir parametre girmeden model oluşturalım 
#Random state burda da belirtildiği için artık best_params_ sonuçları farklı çıkmayacak
mlp_model=MLPRegressor(random_state=5).fit(X_train_scaled,y_train)

y_pred=mlp_model.predict(X_test_scaled)
error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)

#Model tuning

#Hidden layer size =gizli katman sayısı---- 2 değer girdiğimizde 2 katman koy demiş olduk girilen değerlerde o katmandaki noron sayısı
#Verdiğimiz parametrelerden en uygun olanı bulacağız
mlp_params={'alpha':[0.1,0.01,0.02,0.001,0.0001],
            'hidden_layer_sizes':[(10,20),(5,5),(100,100),(120,120)]
            }

mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10).fit(X_train_scaled, y_train)

print(mlp_cv_model.best_params_) #!!!!!!!! alfa bende videodakinden farklı sonuç verdi # ve her çalıştırmamda sonuçlar farklılık gösterebiliyor 


#Final #parametreler videodaki parametrelerden seçildi
mlp_tuned=MLPRegressor(alpha=0.1 , hidden_layer_sizes=(100,100)).fit(X_train_scaled,y_train)
y_pred=mlp_tuned.predict(X_test_scaled)
error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)






























