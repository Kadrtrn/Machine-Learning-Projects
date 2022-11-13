import pandas as pd 

df=pd.read_csv('Advertising.csv')
df=df.iloc[:,1:len(df)]

X=df.drop('sales',axis=1)
Y=df[['sales']]




#Statsmodels ile model kurmak 
import statsmodels.api as sm
#Model nesnemizi oluşturmka için GEREKLİ OLAN  OLS() fonksiyonunu çağırıyoruz
lm=sm.OLS(Y,X)
#fit etmek yani model nesnesini kurmak 
model=lm.fit()

#Modeln özetlerini verir
print(model.summary())



#scikit learn ile model kurmak

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(X,Y)

#Sabiti almak 
print(model.intercept_)

#katsayı değerlerini almak
print(model.coef_)


#Tahmin etmek 
yeni_veri=[[30],[10],[40]]

#T transpoz alıyor
yeni_veri=pd.DataFrame(yeni_veri).T

print(yeni_veri)

#Tahmin
predict=model.predict(yeni_veri)

print(predict)



from sklearn.metrics import mean_squared_error


#HATA KARELER ORTALAMASI MSE
#sol taraf gerçek sağ taraf tahmin edilen değerler 

MSE=mean_squared_error(Y,model.predict(X))
print(MSE)

#HATA KARELER ORTALAMASI KAREKÖKÜ RMSE
import numpy as np
RMSE=np.sqrt(MSE)
print(RMSE)



print('******************')
#MODEL DOĞRULAMA-Model Tuning

#sınama seti
from sklearn.model_selection import train_test_split
#test size yüzde kaçı test için ayrılsın onu söyler  
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=1)


lm=LinearRegression()
model=  lm.fit(X_train,y_train)

#Eğitim Hatası 
train_error=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
print(train_error)

#Test Hatası 
test_error=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
print(test_error)



#Yukarda alınan eğitim ve test setlerinin hangi yüzde 20 ve 80 lik kısımlardan alındığı değiştikce hatalar değişecektir bunun için k katlı cv kullanılır 
#K-Katlı cv  (k katlı crosvaridation)

from sklearn.model_selection import cross_val_score 

#Birinci argüman model  ikinci argüman bagımsız değişkenleri  3. bağımlı değişken 
#cv kaç katlı olacagını scoring ise R kare degerini veya hata kareler ortalamasını elde edebiliriz bize kalmış  burda hata kareler ortalaması alındı 

cv=cross_val_score(model,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print(cv)#Eğitim seti üzerinden 10 tane hata hesaplandı,eğitim seti kendi içinde 10 parçaya bölündü  9 parçayla önce mode kurup dışarda kalan 1 parça tahmin edildi bir hata elde edildi sonra siğer parça dısarsa bırakıldı  9 parçayla model kurulup dışarda kalan parça tahmin edilmeye çalışıldı  ve bu şekilde 10 tane hata elde edildi burdaki değerlerin ortalaması alınacak 
#fonksiyonun kullanımından dolayı değerler eksi çıkıyor önüne eksi koyarak budan kurtulabilriiz

#cv ile elde edilmiş MSE değerimiz 
cv_mean_mse=np.mean(-cross_val_score(model,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
print(cv_mean_mse)

#cv ile RMSE
cv_mean_rmse=np.sqrt(cv_mean_mse)
print(cv_mean_rmse)


































