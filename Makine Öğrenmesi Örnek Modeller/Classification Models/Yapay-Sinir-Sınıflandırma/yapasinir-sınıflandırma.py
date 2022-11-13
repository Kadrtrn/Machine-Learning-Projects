import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV ,cross_val_score
from sklearn.metrics import confusion_matrix , accuracy_score , mean_squared_error, r2_score,roc_auc_score , roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings

warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

df=pd.read_csv('./diabetes.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


y=df['Outcome']
X=df.drop(['Outcome'],axis=1)

X_train ,X_test ,y_train,y_test=train_test_split(
                                                    X,y,
                                                    test_size=0.3,
                                                    random_state=42)

print(df.head())





mlpc_model=MLPClassifier().fit(X_train,y_train)



y_pred=mlpc_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

"""
activation (aktivasyon fonksiyonu): default olarak relu relu regrosyon bölümünde kullanılıyor
relu fonksiyonu dogrusal problemler için kullanılmaktadır bizim burada logistic fonksiyonunu kullanmamız lazım ki 
o da aslında lojistik fonksiyonunda kullanılan sigmoid fonksiyonuna karşılık geliyor değerlendirilebilecek başka aktivasyon fonksiyonlarıda vardır
"""

"""solver modeliimizde kullanılacak olan ağırlıkları optimize etmek için kullanılıcak olan yöntemi ifade ediliyor 
default değeri adam adam daha çok büyük boyutlu veri setlerinde iyi çalışmaktadır veri seti biraz daha küçük olduğunda lbfgs yöntemi daha iyidir
"""

"""Alpha ceaz terimidir """



#Model Tuning

mlpc_params={'alpha':[0.1,0.0001,0.01,5],
             'hidden_layer_sizes':[(3,5),(5,5),(7,7),(100,100)]}

mlpc=MLPClassifier(solver='lbfgs')
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10).fit(X_train, y_train)
#activation şuan relu öncelikle bu şekilde ne olduğuna sonrasıda lojiistik yaptıgımızda neolacagına bir bakalım
relu_best_params=mlpc_cv_model.best_params_
print('activation relu için',relu_best_params)

#Final Model
mlpc_tuned=MLPClassifier(solver='lbfgs',random_state=30,
                         alpha=mlpc_cv_model.best_params_['alpha'],
                         hidden_layer_sizes=mlpc_cv_model.best_params_['hidden_layer_sizes']
                         ).fit(X_train, y_train)


y_pred=mlpc_tuned.predict(X_test)

relu_accuracy=accuracy_score(y_test,y_pred)
print('activation relu için',relu_accuracy)



#Model Tuning activation logistic için

mlpc_params={'alpha':[0.1,0.0001,0.01,5,],
             'hidden_layer_sizes':[(5,5),(7,7),(100,100)]}

mlpc=MLPClassifier(solver='lbfgs',activation='logistic')
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10).fit(X_train, y_train)
#activation şuan relu öncelikle bu şekilde ne olduğuna sonrasıda lojiistik yaptıgımızda neolacagına bir bakalım
logistic_best_params=mlpc_cv_model.best_params_
print('Activation lojistik için',logistic_best_params)

#Final Model
mlpc_tuned=MLPClassifier(solver='lbfgs',activation='logistic',random_state=30,
                         alpha=mlpc_cv_model.best_params_['alpha'],
                         hidden_layer_sizes=mlpc_cv_model.best_params_['hidden_layer_sizes']
                         ).fit(X_train, y_train)


y_pred=mlpc_tuned.predict(X_test)
logistic_accuracy=accuracy_score(y_test,y_pred)
print('Activation lojistik için',logistic_accuracy)




#Değişkenlerin dönüştürülmesi 
"""Kullanmış olduğumuz bütün makine öğrenmesi algoritmaları eğer değişkenler standartlaştırılırsa daha performanslı çalışmaya eğimlidir
Çünkü veri seti içerisinde bulunan aykırılıklar ve benzeri durumlar bu standartlaştırma işlemleriyle bir miktar ortadan kalkabilmektedir
Bu standartlaştırma işlemleri öncesinde ve sonrasında modellerin kullanımı yada modellerin kurulması aşamasında kafa karışıklıkları ortaya çıkarabilmektedir
şöyleki yapay sinir ağları ve daha iyi performans gösterdiğini bildiğimiz derin öğrenme çalışmaları yani görüntü üzerinde yapay sinir ağı kullanılan çalışmalarda 
veri setleri genellikle homojendiir yani değişkenlerin birbirlerine olan benzerlikleri yüksektir örneğin ağaca dayalı yöntemlerde ağaca dayalı algoritmalar 
heterojen veri setlerinde daha iyi çalşır ayrıştırıcılığı yakalamaya çalıştığından dolayı dolayısıyla burdada yine bu yapay sinir ağı
modeli iççin değişken standartlaştırma işlemşi yapılırsa bu  algoritmanın hoşuna gidebilecektir (sonuçlara ne şekilde yansıyacağını şuan kestiriemiyorum dedi videoda  ama standartlaştırma işlemi genelde yapılıyor )

"""



scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

scaler.fit(X_test)
X_test=scaler.transform(X_test)



mlpc_params={'alpha':[0.1,0.0001,0.01,5,],
             'hidden_layer_sizes':[(3,5),(5,5),(7,7),(100,100)]}

mlpc=MLPClassifier(solver='lbfgs',activation='logistic')
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10).fit(X_train, y_train)
#activation şuan relu öncelikle bu şekilde ne olduğuna sonrasıda lojiistik yaptıgımızda neolacagına bir bakalım
print('Activation lojistik için standatlaştırılmış',mlpc_cv_model.best_params_)

#Final Model
mlpc_tuned=MLPClassifier(solver='lbfgs',activation='logistic',random_state=30,
                         alpha=mlpc_cv_model.best_params_['alpha'],
                         hidden_layer_sizes=mlpc_cv_model.best_params_['hidden_layer_sizes']
                         ).fit(X_train, y_train)


y_pred=mlpc_tuned.predict(X_test)

print('Activation lojistik için standartlaştırılmış',accuracy_score(y_test,y_pred))
#accuracy_score artmış oldu
































