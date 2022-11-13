"""Amaç hata kareler toplamını minimize 
eden katsayıları,bu katsayılara bir ceza uygulayarak bulmaktır
"""
import numpy as np 
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt 
from sklearn.linear_model import RidgeCV


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

ridge_model=Ridge(alpha=0.1).fit(X_train,y_train)

#Alfaya göre katsayılar değişiyor unutma 
print(ridge_model.coef_)

lambdalar=10**np.linspace(10,-2,100)*0.5
ridge_model=Ridge()
katsayilar=[]

for i in lambdalar:
    #set_paramas içine veren parametreleri modele ayarlar (Anladığım kadarıyla)
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train,y_train)
    katsayilar.append(ridge_model.coef_)

print(katsayilar)

#Katsayıların durumunu gözlemlleyebilmek için bir grafik oluşturacağız

ax=plt.gca()
#Her bir lambda değerine karşılık gelecek şeiklde lambdaların nasıl oluştuğunu gözlemleyeceğiz
ax.plot(lambdalar,katsayilar)
#Katsayılar birbirinden farklı olabileceğinden dolayı burda bir ölçekleme yapıyoruz 
ax.set_xscale('log')

print(ridge_model)















