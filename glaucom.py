import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载LightGBM模型
model = joblib.load('LGBM.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "TT", "MCV", "PDW", "APTT", "PT", "TC"]

# Streamlit 用户界面
st.title("clinlabomics-based PACG Screening Model")

# 用户输入特征数据
TT = st.number_input("TT:", min_value=0.0, max_value=100.0, value=20.1)
MCV = st.number_input("MCV:", min_value=0.0, max_value=200.0, value=98.0)
PDW = st.number_input("PDW:", min_value=0.0, max_value=100.0, value=13.6)
APTT = st.number_input("APTT:", min_value=0.0, max_value=100.0, value=36.1)
PT = st.number_input("PT:", min_value=0.0, max_value=100.0, value=13.3)
TC = st.number_input("TC:", min_value=0.0, max_value=100.0, value=6.51)

# 将输入的数据转化为模型的输入格式
feature_values = [
    TT, MCV, PDW, APTT, PT, TC
]
features = np.array([feature_values])

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0: normal, 1: PACG)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of PACG. "
            f"The model predicts that your probability of having PACG is {probability:.1f}%. "
        )
    else:
        advice = (
            f"According to our model, you have a low risk of PACG. "
            f"The model predicts that your probability of not having PACG is {probability:.1f}%. "
        )

    st.write(advice)
