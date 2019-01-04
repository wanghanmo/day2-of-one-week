# day2-of-one-week
## 任务说明
在上次任务[day1-of-one-week](https://github.com/wanghanmo/data-for-a-week.git)里，使用了逻辑回归、SVM和决策树三个模型，本次的目的是使用更高级的模型，使用准确率和AUC指标来评价。分别用建随机森林、GBDT、XGBoost和LightGBM这4个模型来对金融数据进行预测。
## 模型构建
### 1.导入需要用到的库
```python
from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import time
```
### 2.对数据进行处理，分开数据和标签
```python
data=pd.read_csv('data_all.csv')
y=data['status']
X=data.drop(['status'],axis=1)
print('the size of X,y:',X.shape,y.shape)
```
```the size of X,y: (4754, 84) (4754,)```
### 3.定义评估函数
```python
def evaluate(pre,y):
    acc=accuracy_score(y,pre)
    model_auc=roc_auc_score(y,pre)
    return acc,model_auc
```
### 4.划分训练集与测试集
训练集与测试集7-3分
```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)```
### 5.模型构建
四个模型分别为随机森林、GBDT、XGBOOST、LightGBM，对这四个模型，设定一致的参数，分别为```learning_rate=0.05```, ```max_depth=3```, ```n_estimators=100```，即相同的初始条件。
```python
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
            learning_rate= 0.05, 
            max_depth=3, 
            n_estimators=100, 
        )),
        ('LGBMClassifier',LGBMClassifier(
            num_leaves=100, 
            max_depth=3,
            learning_rate=0.05,
        ))]
```
### 6.训练模型
```python
for name,model in models:
    print(name,'Start training...')
    starttime=time.clock()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    endtime = time.clock()
    print (name,'所用时间：%.2f'%(endtime - starttime))
    acc,model_auc=evaluate(preds,y_test)
    print(name,'accuracy_score,roc_auc_score：',acc,model_auc)
```
### 7.实验结果
模型名称|所用时间|accuracy_score|AUC
-------|-------|--------------|---
RandomForestClassifier|0.28|0.7666433076384023|0.5436084420936225
GradientBoostingClassifier|0.97|0.7841625788367204|0.6339029555673791
XGBClassifier|1.36|0.7897687456201822|0.6376482739194391
LGBMClassifier|0.25|0.7904695164681149|0.6371918458472869
### 8.结论
由实验结果可以看出，LightGBM有着最快的运行速度，并和XGBOOST有着相当的准确率。
## 参考
- sklearn官方英文文档：https://scikit-learn.org/stable/index.html
- sklearn中文版文档：http://sklearn.apachecn.org/#/
- xgboost官方英文文档：https://xgboost.readthedocs.io/en/latest/
- lightgbm英文官方文档：https://lightgbm.readthedocs.io/en/latest/
