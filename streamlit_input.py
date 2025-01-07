import pickle
import pandas as pd
import streamlit as st
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# r2=r2_score(y_test,y_predict)
with open('rfr_model.pkl','wb') as f:
    pickle.dump(rfr,f)

print('saved')

with st.form('user_inputs'):
    age=st.number_input('age',min_value=0)
    sex=st.radio('sex',options=['male','female'])
    bmi=st.number_input('BMI',min_value=0.0)

    children=st.number_input('number of children',min_value=0,step=1)
    smoke=st.radio('smoking or not',['yes','no'])
    region=st.selectbox('region',('northeast','southeast','northeast','northwest'))
    submitted=st.form_submit_button('prediced charges')

if submitted:
    form_data=[age,sex,bmi,children,smoke,region]
    st.write('user information as follows:')
    st.text(form_data)

    sex_male,sex_female=0,0
    if sex=='male':
        sex_male=1
    elif sex=='female':
        sex_female=1

    smoke_yes, smoke_no=0,0
    if smoke=='yes':
        smoke_yes=1
    elif smoke=='no':
        smoke_no=1

    region_northeast,region_southeast, region_northwest,region_southwest=0,0,0,0
    if region =='northeast':
        region_northeast=1
    elif region=='southeast':
        region_southeast=1
    elif region=='northwest':
        region_northwest=1
    elif region=='southwest':
        region_southwest=1

    st.write('transformed user information:')
    format_data=[age,bmi,children,sex_male,sex_male,smoke_no,smoke_yes,
                 region_northeast,region_southeast,region_northwest,region_southwest]
    st.text(format_data)

    with open('/Users/renwei/pythonProject4/10_10/new_doc/rfr_model.pkl','rb') as f:
        rfr_model=pickle.load(f)

    if submitted:
        format_data_df=pd.DataFrame(data=[format_data], columns=rfr_model.feature_names_in_)
        predict_result=rfr_model.predict(format_data_df)[0]

        st.write('predicted charges:', round(predict_result,2))