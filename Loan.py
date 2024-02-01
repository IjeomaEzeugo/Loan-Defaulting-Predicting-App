import streamlit as st
import pandas as pd
import numpy as np
from pickle import dump
import pickle

model = pickle.load(open('rg_model.pkl','rb'))
Scaler = pickle.load(open('reg_scaler.pkl','rb'))
encoder= pickle.load(open('reg_enc.pkl','rb'))

def data_table():
    C1, C2, C3, = st.columns(3)

    with C1:
        AC = st.number_input("What is the Asset Cost", step=500)
        BI = st.number_input('BRANCH_ID', step =1)
        SI = st.number_input('SUPPLIER_ID', step = 1)
        MI = st.number_input('MANUFACTURER_ID',step=1)
        CPI = st.number_input('CURRENT_PINCODE_ID', step=1)

    with C2:
        MAF = st.selectbox("Is there a mobile number?",["YES","NO"])
        if MAF=="Yes":
            MAF1 = 1
        else:
            MAF1 = 0

        PCB = st.number_input('What is the current balance',step=100)
        PI= st.number_input("Primary Installment",step=100)
        AAA = st.number_input('How old is the account(Months)',step=1)
        CHL= st.number_input("How long is the credit history(Month)", step=1)
        NOI = st.number_input("Number of inquiries",step=1)

    with C3:
        YR=st.number_input("Year of Birth", step=1)
        MTH = st.selectbox("Month of Birth",range(1,13))
        D= st.selectbox("Day of Birth", range(1,32))
        PD= st.number_input("Percentage Deposited", step = 1)
        ET = st.selectbox("Employment Type",['Salaried','Self Employed'])

    feat = np.array([AC,BI,SI,MI,CPI,MAF1,PCB,PI,AAA,CHL,NOI,
                     YR,MTH,D,PD,ET]).reshape(1,-1)
    
    cat =  ['ASSET_COST', 'BRANCH_ID', 'SUPPLIER_ID', 'MANUFACTURER_ID',
      'CURRENT_PINCODE_ID','MOBILENO_AVL_FLAG', 'PRI_CURRENT_BALANCE', 'PRIMARY_INSTAL_AMT',
      'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH','NO_OF_INQUIRIES','Year',
      'Month', 'Day', 'Percent Deposit','EMPLOYMENT_TYPE']
    
    table = pd.DataFrame(feat,columns=cat)
    return table

def process(df):
     enc_data = pd.DataFrame(encoder.transform(np.array(df[['EMPLOYMENT_TYPE']])).toarray(),\
                            columns=encoder.get_feature_names_out(['EMPLOYMENT_TYPE']))
     
     df = pd.concat([df, enc_data], axis=1)
     
     df.drop("EMPLOYMENT_TYPE",axis=1,inplace=True)

     col = df.columns

     df = Scaler.transform(df)

     df = pd.DataFrame(df,columns=col)

     df.drop('PRIMARY_INSTAL_AMT',axis=1,inplace=True)

     return df


frame = data_table()

if st.button('Predict'):
    frame2 =process(frame)
    pred = model.predict(frame2)
    st.write(round(pred[0],0))