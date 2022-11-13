import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
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
#Pregnancies hamilelik sayısı 
#Glucose glikoz 
#BMI vucüt kitle endeksi
#print(df.head())




#Model ve Tahmin

#print(df['Outcome'].value_counts())

#print(df.describe().T)

y=df['Outcome']
X=df.drop(['Outcome'],axis=1)

#Boş bir şekilde oluşturabiliriz veya liblinear  diyebiliriz çünkü
#logistik regresyonda katsayıları bulmak adına kullanılabilen birden fazla 
#minimizasyon yaklaşımı var yani gerçek değerler ile tahmin edilen değerler 
#arasındaki farkların karelerinin toplamımı ifade etmek adına bazı birbirinden farklı yöntemler var 
#hatta ridge ve lassodan alışık olduğumuz düzenlileştirme yöntemlerininde işin içine katıldığı 
#bazı katsayı bulma yöntemleri var biz burda liblinear kullanıyoruz
loj_model=LogisticRegression(solver='liblinear').fit(X,y)

print(loj_model.intercept_)
print(loj_model.coef_)

#!!!!!!!! train test ayırımı burda yapılmadı unutma 

y_pred=loj_model.predict(X)


#Karmaşıklık matrisi
print(confusion_matrix(y, y_pred))

#Doğruluk oranı bulma 
print(accuracy_score(y, y_pred))

#Classification_report
#Detaylı bir sınıflandırma raporu 
print(classification_report(y, y_pred))



#Tahminleri 1 ve 0 olarak değilde direkt olasılık olarak almak 
#Sağdakiler 1 soldakiler 0
print(loj_model.predict_proba(X)[:10])


#Model Doğrulama 


X_train ,X_test ,y_train ,y_test =train_test_split(X,
                                                   y,
                                                   test_size=0.3,
                                                   random_state=42)

loj_model=LogisticRegression(solver='liblinear').fit(X_train,y_train)

y_pred=loj_model.predict(X_test)

print(accuracy_score(y_test, y_pred))


print('**************')
#K katlı çapraz doğrulama
#Ortalamayı aldığımızda test setine ilişkin daha doğru bir test hatası değerine erişmiş olduk 
cv=cross_val_score(loj_model, X_test,y_test,cv=10).mean()
print(cv)
























