import pandas as pd
import joblib
import numpy as np
import streamlit as st

st.title("Medical Charges Prediction App")
st.caption("This app predicts the Medical charges for an insurance Company based on the input")

#load the saved model
model = joblib.load('best_model.pkl')


#get user input
age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=0.0, format='%f')
children = st.number_input('Children', min_value=0, step=1)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Create a DataFrame from the inputs
input_data = {
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region],
}

user_input_df = pd.DataFrame(input_data)

model_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest', 'age_squared', 'obesity' ,'smoker_obesity_interaction']

# Create a DataFrame with zeros for all model columns
df_template = pd.DataFrame(0, index=np.arange(1), columns=model_columns)

# Update the template with the user input
for column in user_input_df.columns:
    #add a suffix and set the appropriate dummy column If the column is categorical
    if user_input_df[column].dtype == 'object':
        dummy_name = f"{column}_{user_input_df.iloc[0][column]}"
        if dummy_name in df_template.columns:
            df_template[dummy_name] = 1
    else:
        # For numerical columns, just copy the value
        df_template[column] = user_input_df[column]


# Adding a non-linear transformation of the 'age' feature
df_template['age_squared'] = df_template['age']**2


#Define obesity as BMI > 30
df_template['obesity'] = (df_template['bmi'] > 30).astype(int)
# st.write(df_with_dummies)

#Create an interaction term for smoker and obesity
df_template['smoker_obesity_interaction'] = df_template['smoker_yes'] * df_template['obesity']

st.write(df_template)
if st.button('Predict'):
    prediction = model.predict(df_template)
    st.write(f'The Predicted Medical Charges is: {prediction}')


