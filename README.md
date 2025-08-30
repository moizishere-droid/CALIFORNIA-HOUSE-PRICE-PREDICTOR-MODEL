🏡 California House Price Prediction: End-to-End ML Project

As part of my machine learning journey, I worked on a **California House Price Prediction project**, building, tuning, and deploying a regression model to predict housing prices.
The goal was not just to train a model, but to experience the complete ML lifecycle—from model selection to deployment.

🔍 Project Overview

Predicting house prices is a classic machine learning problem that involves:

* Understanding and processing structured data.
* Testing multiple regression algorithms.
* Applying hyperparameter tuning to improve accuracy.
* Deploying the best model with an interactive frontend.

🧑‍💻 Steps I Followed

1. Data Preparation

I started with the **California housing dataset**, which contains features like:

* Median income
* Number of rooms
* Population
* House age
* Location (latitude, longitude)

I split the dataset into **training and test sets** for fair evaluation.

2. Model Selection & Performance

I experimented with multiple regression models and measured their **R² scores** before and after hyperparameter tuning.

| Model                       | Base Mean R² | Base Std |
| --------------------------- | ------------ | -------- |
| LightGBM (LGBM)             | 0.6119       | 0.0072   |
| HistGradientBoosting (HGBR) | 0.8361       | 0.0079   |
| XGBoost (XGB)               | 0.8323       | 0.0082   |
| Extra Trees (ETR)           | 0.8129       | 0.0081   |

📊 **Observation:
3. Hyperparameter Tuning

I applied **hyperparameter tuning** (using techniques like Optuna) to maximize model performance.

| Model                       | Tuned Mean R² | Tuned Std |
| --------------------------- | ------------- | --------- |
| LightGBM (LGBM)             | 0.8588        | 0.0045    |
| HistGradientBoosting (HGBR) | 0.8468        | 0.0078    |
| XGBoost (XGB)               | 0.8572        | 0.0066    |
| Extra Trees (ETR)           | 0.8045        | 0.0089    |


* ✅ **Best Model:** LightGBM
* ✅ **Final R² Score:** 0.8533

4. Saving the Model

I used **Joblib** to save the trained LightGBM model. This allows the model to be reused later without retraining.

'''
python
import joblib
joblib.dump(best_lgbm_model, "lgbm_model.pkl")
```

5. Building a Frontend with Streamlit

I created a **Streamlit web application** where users can input housing features and get **predicted house prices instantly**.

Features of the app:

* Clean and simple UI
* User-friendly input fields
* Real-time predictions

```python
import streamlit as st
import numpy as np
import joblib

loaded_model = joblib.load("lgbm_model.pkl")

f1 = st.slider("Median Income (10k$)", 0.5, 15.0, 3.5)
f2 = st.slider("House Age (years)", 1, 52, 29)
# ... other sliders

features = np.array([[f1, f2, ...]])
if st.button("Predict"):
    prediction = loaded_model.predict(features)[0]
    st.success(f"🏠 Estimated Median House Value: ${prediction * 100000:,.2f}")
```

6. Deployment

I deployed the Streamlit app on **Streamlit Cloud**, making it accessible online.

👉 [California House Price Predictor](https://california-house-price-predictor-model-jjl8vkggrm79pmjlqq6rkt.streamlit.app/)

Now, anyone can predict California house prices in real-time using the interactive web app.

🚀 Key Learnings

* Improved my understanding of **regression models**.
* Learned the importance of **hyperparameter tuning** for performance.
* Gained experience in **model deployment** using Streamlit Cloud.
* Built an **end-to-end ML pipeline**, from training to deployment.

📌 Conclusion

This project gave me hands-on experience in applying machine learning techniques to solve real-world problems.

Even though I’m based in **Karachi, Pakistan**, working on a dataset from California helped me understand how ML can generalize across regions and domains.
