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


#Model ve Tahmin
#kernel rbf dogrusal olmayan linear dogrusal olan model
svm_model=SVC(kernel='linear').fit(X_train,y_train)

y_pred=svm_model.predict(X_test)

print(accuracy_score(y_test, y_pred))




#Model Tuning

svm=SVC()
svm_params= {'C':np.arange(1,3),
             'kernel':['linear','rbf']}

svm_cv_model=GridSearchCV(svm,svm_params,cv=5).fit(X_train,y_train)

print(svm_cv_model.best_score_)
print(svm_cv_model.best_params_)
print(svm_cv_model.best_params_['kernel'])
svm_tuned=SVC(kernel=svm_cv_model.best_params_['kernel'],
              C=svm_cv_model.best_params_['C']).fit(X_train,y_train)

y_pred=svm_tuned.predict(X_test)

print(accuracy_score(y_test, y_pred))

























