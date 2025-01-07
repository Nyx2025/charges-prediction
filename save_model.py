import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

insurance_df=pd.read_csv(f'/Users/renwei/Desktop/insurance.csv',encoding='utf-8')
# print(insurance_df.info)

output=insurance_df['charges']
features=insurance_df[['age','sex','bmi','children','smoker','region']]
features=pd.get_dummies(features).astype(int)
# print(features.head())

x_train,x_test,y_train,y_test=train_test_split(features,output,train_size=0.8)
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
y_predict=rfr.predict(x_test)

r2=r2_score(y_test,y_predict)
with open('rfr_model.pkl','wb') as f:
    pickle.dump(rfr,f)

print('saved')