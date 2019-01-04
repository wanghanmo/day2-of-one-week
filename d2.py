from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import time

def evaluate(pre,y):
    acc=accuracy_score(y,pre)
    model_auc=roc_auc_score(y,pre)
    return acc,model_auc


data=pd.read_csv('data_all.csv')

y=data['status']
X=data.drop(['status'],axis=1)
print('the size of X,y:',X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)


models=[('RandomForestClassifier',RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            criterion='gini'
        )),
        ('GradientBoostingClassifier',GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.05,  
            max_depth=3,  
        )),
        ('XGBClassifier',XGBClassifier(
            ilent=0 ,
            learning_rate= 0.05, 
            max_depth=3, 
            n_estimators=100, 
        )),
        ('LGBMClassifier',LGBMClassifier(
            num_leaves=150, 
            objective='binary',
            max_depth=3,
            learning_rate=0.05,
            max_bin=100
        ))]
for name,model in models:
    print(name,'Start training...')
    starttime=time.clock()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    endtime = time.clock()
    print (name,'所用时间：%.2f'%(endtime - starttime))
    acc,model_auc=evaluate(preds,y_test)
    print(name,'accuracy_score,roc_auc_score：',acc,model_auc)
