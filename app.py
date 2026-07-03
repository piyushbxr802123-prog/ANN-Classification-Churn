import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- Load model weights (NO tensorflow needed) ----------
data = np.load("model_weights.npz")
W = [data[k] for k in data.files]   # W[0],W[1] = layer1 w,b | W[2],W[3] = layer2 | W[4],W[5] = output

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X):
    """Manual forward pass of the ANN"""
    a = relu(X @ W[0] + W[1])       # hidden layer 1
    a = relu(a @ W[2] + W[3])       # hidden layer 2
    a = sigmoid(a @ W[4] + W[5])    # output layer
    return a
# ----------------------------------------------------------------

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure', 0, 10)
num_of_products=st.slider('Number of Products', 1, 4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],           # ✅ fixed
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]    # ✅ fixed
})

geo_encoded= label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled=scaler.transform(input_data)

prediction = predict(input_scaled)
prediction_proba = float(prediction[0][0])
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
