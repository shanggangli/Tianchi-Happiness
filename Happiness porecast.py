import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error #RMSE
import xgboost as xgb
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB #贝叶斯
data_train=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\happiness_train_complete.csv',encoding='gbk'))
data_test=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\happiness_test_complete.csv',encoding='gbk'))
data_testid=data_test['id']
#print('train:',data_train.info())
#print(data_train.shape)
#print('test:',data_test.info())
#print(data_test.shape)
#print(data_train.info(verbose=True,null_counts=True))

#缺失值
'''temp=data_train.isnull().sum()
print(temp[temp!=0])'''

'''data_train['happiness'].value_counts().plot(kind='bar')
plt.show()'''
data_train['happiness']=data_train['happiness'].map(lambda x:3 if x==-8 else x)
#print(data_train['happiness'].value_counts())
# 合并训练数据和测试数据
data = pd.concat([data_train,data_test],axis=0,ignore_index=True,sort=False)
#print(data.shape)
#print(data.info(verbose=True,null_counts=True))
#处理时间特征
data['survey_time']=pd.to_datetime(data['survey_time'])
data['year']=data['survey_time'].dt.year
data['month']=data['survey_time'].dt.month
data['hour']=data['survey_time'].dt.hour
# 问卷调查时的年龄
data['survey_year']=data['year']-data['birth']

#print(data['birth'].describe())
def birth_split(x):
    if 1920<=x<=1930:
        return 0
    elif  1930<x<=1940:
        return 1
    elif  1940<x<=1950:
        return 2
    elif  1950<x<=1960:
        return 3
    elif  1960<x<=1970:
        return 4
    elif  1970<x<=1980:
        return 5
    elif  1980<x<=1990:
        return 6
    elif  1990<x<=2000:
        return 7
data['birth']=data['birth'].map(birth_split)
data=data.drop(["edu_other"], axis=1)
data=data.drop(["survey_time"], axis=1)
data['s_birth']=data['s_birth'].map(birth_split)

# 收入分组
def income_cut(x):
    if x <= 0 :
        return 0
    elif 0 < x <=1200:
        return 1
    elif 1200 < x <=10000:
        return 2
    elif 10000 < x <= 24000:
        return 3
    elif 24000 < x <=40000:
        return 4
    elif 40000 <= x:
        return 5
data["income"] = data["income"].map(income_cut)
data['s_income']=data['s_income'].map(income_cut)

# 结婚年龄
data['marital_now']=2015-data['marital_now']
data['marital_now']=data['marital_now'].map(lambda x :0 if x<=0 else x)

# 入党年龄

#填充缺失值
data['edu_status']=data['edu_status'].fillna(5)
data["edu_yr"]=data["edu_yr"].fillna(-2)
data["property_other"]=data["property_other"].map(lambda x:0 if pd.isnull(x)  else 1)
data["join_party"]=data["join_party"].map(lambda x:0 if pd.isnull(x)  else 1)
#print(data["hukou_loc"].value_counts())
data["hukou_loc"]=data["hukou_loc"].fillna(1)
#print(data['social_neighbor'].value_counts())
data['social_neighbor']=data['social_neighbor'].fillna(8)
data['social_friend']=data['social_friend'].fillna(8)
data["work_status"]=data["work_status"].fillna(0)
data["work_yr"]=data["work_yr"].fillna(0)
data["work_type"]=data["work_type"].fillna(0)
data["work_manage"]=data["work_manage"].fillna(0)
data["family_income"]=data["family_income"].fillna(-2)
data["invest_other"]=data["invest_other"].map(lambda x:0 if pd.isnull(x)  else 1)
data["minor_child"]=data["minor_child"].fillna(0)
data["marital_1st"]=data["marital_1st"].fillna(0)
data["s_birth"]=data["s_birth"].fillna(0)
data["marital_now"]=data["marital_now"].fillna(0)
data["s_edu"]=data["s_edu"].fillna(0)
data["s_political"]=data["s_political"].fillna(0)
data["s_hukou"]=data["s_hukou"].fillna(0)
data["s_income"]=data["s_income"].fillna(0)
data["s_work_exper"]=data["s_work_exper"].fillna(0)
data["s_work_status"]=data["s_work_status"].fillna(0)
data["s_work_type"]=data["s_work_type"].fillna(0)

#print(data.info(verbose=True,null_counts=True))

data=data.drop(['id'],axis=1)
# 特征相关性分析
'''plt.figure(figsize=(25,10))
Corr=data.corr()['happiness'].sort_values(ascending=False)
Corr.plot(kind='bar')
print(Corr)
plt.show()'''
data=data.drop(['work_status','work_manage','property_other','invest_0'],axis=1)
train_data=data.loc[:7999,:]
test_data=data.loc[8000:,:]
train_data=pd.DataFrame(train_data)
test_data=pd.DataFrame(test_data)


target=train_data['happiness']
train_data=train_data.drop(['happiness'],axis=1)
test_data=test_data.drop(['happiness'],axis=1)
'''print('#####################')
print(train_data.shape)
print(test_data.shape)
print(train_data.info(verbose=True,null_counts=True))
print(test_data.info(verbose=True,null_counts=True))'''

# 训练模型
X_train,X_test,y_train,y_test=train_test_split(train_data,target,random_state=0,train_size=0.7)
print(",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
print(",训练数据标签:",y_train.shape,
     ',测试数据标签:',y_test.shape )

# LR 0.69
'''LR=LogisticRegression(C=1.0)
LR.fit(X_train,y_train)
pred=LR.predict(X_test)
MSE=mean_squared_error(y_test,pred)
print(MSE)'''

# xgb 0.493
model_xgb= xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2);
model_xgb.fit(X_train,y_train)
xgb_pred=model_xgb.predict(X_test)
MSE=mean_squared_error(y_test,xgb_pred)
print(MSE)

# test
xgb_predtest=model_xgb.predict(test_data)
result=pd.DataFrame({'id':data_testid,'happiness':xgb_predtest})
#result.to_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\1.csv',index=False)

# 贝叶斯 1.25
'''NB=BernoulliNB()
NB.fit(X_train,y_train)
NB_predict=NB.predict(X_test)
MSE=mean_squared_error(y_test,NB_predict)
print(MSE)'''

# knn 0.909
'''from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
MSE=mean_squared_error(y_test,knn_pred)
print(MSE)'''
