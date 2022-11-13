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



import random 

random_number=random.randint(0, 100)
rf_model=RandomForestClassifier(random_state=random_number).fit(X_train, y_train)


y_pred=rf_model.predict(X_test)

print(accuracy_score(y_test, y_pred))



#Model_tuning

#max_features bölünmelerde göz önünde bulundurulması gereken değişken sayısı
rf=RandomForestClassifier()
rf_params={
    'n_estimators':[500,600,700],
    'max_features':[7,8],
    'min_samples_split':[5,7]
    }


rf_cv_model=GridSearchCV(rf, rf_params,cv=10).fit(X_train,y_train)


print(rf_cv_model.best_params_)


#Final Model

rf_tuned=RandomForestClassifier(n_estimators=rf_cv_model.best_params_['n_estimators'],
                                max_features=rf_cv_model.best_params_['max_features'],
                                min_samples_split=rf_cv_model.best_params_['min_samples_split'],
                                random_state=random_number
                                ).fit(X_train, y_train)


y_pred=rf_tuned.predict(X_test)

print(accuracy_score(y_test, y_pred))

#Değişkenlerin önem düzeyi ile ilgili bilgi
print(rf_tuned.feature_importances_)


#Değişken Önem Düzeyleri 

feature_imp=pd.Series(rf_tuned.feature_importances_,
                      index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('Değinken Önem Skorları ')
plt.ylabel=('Değişkenler')
plt.title('Değişken Önem Düzeyleri')

plt.show()




















