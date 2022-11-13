import numpy as np
import pandas as pd 
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt 
from sklearn.linear_model import RidgeCV,LassoCV


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

lasso_model=Lasso().fit(X_train, y_train)

#print(lasso_model)

#print(lasso_model.intercept_)

#print(lasso_model.coef_)


#Farklı lambda değerlerine karşılık katsayılar 

lasso=Lasso()
coefs=[]

alphas=np.random.randint(0,1000,10)

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)
    
    
ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale('log')


#Tahmin

print(lasso_model.predict(X_train)[0:5])
print(lasso_model.predict(X_test)[0:5])

y_pred=lasso_model.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

#Bağımsız değişkenlece bağımlı değişkendeki değişikliğin açıklanma yüzdesidir 
print(r2_score(y_test, y_pred))

#Model Tuning
lasso_cv_model=LassoCV(cv=10,max_iter=100000).fit(X_train,y_train)
print(lasso_cv_model.alpha_)


lasso_tuned=Lasso().set_params(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)

#Üst satır şu şekilde de olabilir 
#lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred=lasso_tuned.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

#Yukarıda lasso_tuned içine aranacak bir alfa uzayı vermedik  kendisi bize bir alfa önerdi (bu alfa çevresinde yeni bir alfa aranabilir )
#aşağıda bir alfa uzayı vererek hatayı tekrar gözlemleyeceğiz 
alphas=10**np.linspace(10,-2,100)*0.5

lasso_cv_model=LassoCV(alphas=alphas,cv=10,max_iter=100000).fit(X_train,y_train)
print(lasso_cv_model.alpha_)


lasso_tuned=Lasso().set_params(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)

#Üst satır şu şekilde de olabilir 
#lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred=lasso_tuned.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


#Değişkenlerin katsayılarını getirelim 
coefss=pd.Series(lasso_tuned.coef_,index=X_train.columns)
print(coefss)























