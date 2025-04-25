import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import streamlit
import joblib
import numpy
import pandas
import shap
import matplotlib


model_path = 'model_catbooswhy1112.pkl'
model = joblib.load(model_path)
import os


# Define feature names
feature_names = [
    "HR", "P", "albumin", "glucose", "Ka", "NT.proBNP", "CRP", "INR"
]

# Streamlit 名字
st.title("Acute coronary syndrome 3-year mortality predictor")
model_path1 = 'min_max_RFrfrfrfrrwhy112.pkl'
min_max_scaler = joblib.load(model_path1 )
# User inputs
import streamlit as st

HR = st.number_input("HR:", min_value=40.0, max_value=140.0, value=80.0)
albumin = st.number_input("Albumin:", min_value=20.0, max_value=50.0, value=30.0)#
P= st.number_input("P :", min_value=0.0, max_value=3.0, value=1.0)
Ka = st.number_input("Ka:", min_value=2.0, max_value=7.0, value=3.0)
INR = st.number_input("INR:", min_value=0.5, max_value=3.0, value=2.0)
glucose = st.number_input("Glucose:", min_value=0.5, max_value=20.0, value=3.0)#
NT_proBNP = st.number_input("NT.proBNP:", min_value=0.1, max_value=2000.0, value=20.0) #
CRP = st.number_input("CRP:", min_value=0.1, max_value=30.0, value=20.0) #

# Process inputs and make predictions
feature_values = [HR, albumin, P, glucose, INR, Ka, CRP ,NT_proBNP]
features_scaled = np.array([feature_values])
# 使用加载的 MinMaxScaler 对用户输入的数据进行归一化
features = min_max_scaler.transform(features_scaled)
# Load background data for SHAP
background_data_path =  'model_cRFrrfrfrwhy112.csv'
background_data = pd.read_csv(background_data_path)

# 确保选择特定的列进行缩放
feature_names = [ "HR" ,"P","albumin","glucose","Ka","NT.proBNP", "CRP", "INR"]
background_data_scaled = min_max_scaler.transform(background_data[feature_names].values)

if st.button("Predict"):
    # Predict probabilities
    predicted_proba = model.predict_proba(features)[0]
    # 只获取类别为 0 的预测概率
    probability_class_0 = predicted_proba[0] * 100

    # Display prediction results for class 0
   # st.write(f"**Probability of Class 0:** {probability_class_0:.1f}%")

    # SHAP Explanation for class 0
    st.subheader(
        f"Based on feature values,the predicted 3-year mortality rate of acute coronary syndrome is: **{probability_class_0:.1f}%**")
    explainer_shap = shap.KernelExplainer(model.predict_proba, background_data_scaled)
    shap_values = explainer_shap(pd.DataFrame(features, columns=feature_names))

    # 获取 expected_value 和 shap_values 中类别为 0 的部分
    if isinstance(explainer_shap.expected_value, list) or isinstance(explainer_shap.expected_value, np.ndarray):
        expected_value = explainer_shap.expected_value[0]
        shap_values_for_class_0 = shap_values.values[0, :, 0]
    else:
        expected_value = explainer_shap.expected_value
        shap_values_for_class_0 = shap_values.values[0, :]

    shap.force_plot(expected_value, shap_values_for_class_0, pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True)
    plt.savefig("shap_force_plot_class_0.png", bbox_inches='tight', dpi=500)
    st.image("shap_force_plot_class_0.png", caption='')
