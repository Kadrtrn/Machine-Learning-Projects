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




from xgboost import XGBClassifier

xgb_model=XGBClassifier().fit(X_train,y_train)

#?xgb_model


y_pred=xgb_model.predict(X_test)


print(accuracy_score(y_test, y_pred))


#Model Tuning

xgb=XGBClassifier()

#subsamplme göz önünde bulundurulacak olan örneklem oranını ifade ediyor
xgb_params={
        'n_estimators':[200,500,600],
        'subsample':[0.6,0.8],
        'max_depth':[5,7],
        'learning_rate':[0.1,0.01,0.001]
    }


xgb_cv_model=GridSearchCV(xgb, xgb_params,cv=10).fit(X_train,y_train)
print(xgb_cv_model.best_params_)

print(xgb_cv_model.best_score_)


#Final model 
xgb_tuned=XGBClassifier(learning_rate=xgb_cv_model.best_params_['learning_rate'],
                        subsample=xgb_cv_model.best_params_['subsample'],
                        n_estimators=xgb_cv_model.best_params_['n_estimators'],
                        max_depth=xgb_cv_model.best_params_['max_depth']
                        ).fit(X_train,y_train)


y_pred=xgb_tuned.predict(X_test)

print(accuracy_score(y_test, y_pred))







#Değişkenlerin önem düzeyi ile ilgili bilgi
print(xgb_tuned.feature_importances_)


#Değişken Önem Düzeyleri 

feature_imp=pd.Series(xgb_tuned.feature_importances_,
                      index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('Değinken Önem Skorları ')
plt.ylabel=('Değişkenler')
plt.title('Değişken Önem Düzeyleri')

plt.show()





















