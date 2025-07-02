import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the pre-trained model and encoders
model = tf.keras.models.load_model('src/customer_churn_model.h5')

## load the encoders and scaler

with open('/Users/balmukundmishra/Desktop/2025-Learning/ANN_Classification/src/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('/Users/balmukundmishra/Desktop/2025-Learning/ANN_Classification/src/one_hot_encode_contr.pkl', 'rb') as f:
    one_hot_encoder_contr = pickle.load(f)

with open('/Users/balmukundmishra/Desktop/2025-Learning/ANN_Classification/src/one_hot_encode_subs.pkl', 'rb') as f:
    one_hot_encoder_subs = pickle.load(f)


with open('/Users/balmukundmishra/Desktop/2025-Learning/ANN_Classification/src/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction')

# Age,Gender,Tenure,Usage Frequency,Support Calls,Payment Delay,Subscription Type,Contract Length,Total Spend,Last Interaction,Churn
# Age,Gender,Tenure,Usage Frequency,Support Calls,Payment Delay,Subscription Type,Contract Length,Total Spend,Last Interaction
# these are the input data i am using
# lets create a form to take input from the user
Age = st.selectbox('Age', options=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
Gender = st.selectbox('Gender', options=['Male', 'Female'])
Tenure = st.number_input('Tenure (in months)', min_value=0, max_value=100, value=12)
Usage_Frequency = st.number_input('Usage Frequency', min_value=0, max_value=1000, value=100)
Support_Calls = st.number_input('Support Calls', min_value=0, max_value=10, value=2)
Payment_Delay = st.number_input('Payment Delay (in days)', min_value=0, max_value=30, value=5)
Subscription_Type = st.selectbox('Subscription Type', options=['Basic', 'Standard', 'Premium'])
Contract_Length = st.selectbox('Contract Length', options=['Monthly', 'Quarterly', 'Anually'])
Total_Spend = st.number_input('Total Spend (in $)', min_value=0, max_value=1000, value=100)
Last_Interaction = st.number_input('Last Interaction', min_value=0, max_value=1000, value=100)

# prepare the input data

input_data = pd.DataFrame({
    'Age': [Age],
    'Gender' : [label_encoder.transform([Gender])[0]],
    'Tenure': [Tenure],
    'Usage Frequency': [Usage_Frequency],
    'Support Calls': [Support_Calls],
    'Payment Delay': [Payment_Delay],
    'Total Spend': [Total_Spend],
    'Last Interaction': [Last_Interaction]
})

df = pd.DataFrame(one_hot_encoder_contr.transform([[Contract_Length]]).toarray(), 
                  columns=one_hot_encoder_contr.get_feature_names_out(['Contract Length']))
df1 = pd.DataFrame(one_hot_encoder_subs.transform([[Subscription_Type]]).toarray(), 
                   columns=one_hot_encoder_subs.get_feature_names_out(['Subscription Type']))


input_data = pd.concat([input_data.reset_index(drop=True), df, df1], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the churn probability
if st.button('Predict Churn'):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    
    # Display the result
    st.write(f'Churn Probability: {churn_probability:.2f}')
    
    if churn_probability > 0.5:
        st.warning('The customer is likely to churn.')
    else:
        st.success('The customer is likely to stay.')


