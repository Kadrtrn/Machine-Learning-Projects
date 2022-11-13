import pandas as pd 
df=pd.read_csv('Advertising.csv')
df=df.iloc[:,1:len(df)]

print(df.head())
print(df.info())
import seaborn as sns 


#sns.jointplot(x='TV' ,y='sales',data=df,kind='reg')

from sklearn.linear_model import LinearRegression

X=df[['TV']]
Y=df[['sales']]

reg= LinearRegression()
model=reg.fit(X,Y)

print(model)
print(dir(model))

#model.intercept_  sabitimiz 
print(model.intercept_)
#model.coef_   buda katsayımız
print(model.coef_)

#rkare Bağımlı değişkendeki değişikliğin, bağımsız değişkenlerce açıklanma yüzdesidir
print(model.score(X,Y)) #yani bu örneğimizde satışlardaki değişikliğin yaklaşık yüzde 60'ı bağımsız değişkence tani 'TV' ile açıklanmaktadır 

import matplotlib.pyplot as plt
"""
g=sns.regplot(df['TV'],df['sales'],ci=None,scatter_kws={'color':'r','s':9})
g.set_title('Model Denklemi: Sales = 7,03 + 0,5*TV')
g.set_ylabel('Satış Sayısı')
g.set_xlabel('TV Harcamaları')
plt.xlim('-10,310')
plt.ylim(bottom=0)
"""

#tahmşn değerini skelarn kullanarak bulmak 

#bagımsız değişkeni içeri satıyoruz
predict=model.predict([[165]])
print(predict)

#Çoklu bir şekilde tahmin etmek
multiple_data=[[5],[15],[30]]
multiple_predict=model.predict(multiple_data)
print(multiple_predict)


















