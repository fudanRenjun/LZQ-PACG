import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载随机森林模型
model = joblib.load('LGBM.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "TT", "MCV", "PDW", "APTT", "PT", "TC"]

# Streamlit 用户界
st.title("clinlabomics-based PACG Screening Model")

# 用户输入特征数据
TT = st.number_input("TT:", min_value=0.0, max_value=100.0, value=0)
MCV = st.number_input("MCV:", min_value=0.0, max_value=100.0, value=0)
PDW = st.number_input("PDW:", min_value=0.0, max_value=100.0, value=0)
APTT = st.number_input("APTT:", min_value=0.0, max_value=100.0, value=0)
PT = st.number_input("PT:", min_value=0.0, max_value=100.0, value=0)
TC = st.number_input("TC:", min_value=0.0, max_value=100.0, value=0)

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

    # 计算并显示SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测结果生成并显示SHAP force plot
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    # 保存SHAP图并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
