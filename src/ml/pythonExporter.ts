export const PYTHON_TRAIN_SCRIPT = `"""
Customer Classification using AdaBoost - Model Training Script
College AI/ML Project
Objective: Train baseline Decision Tree vs AdaBoost Classifier on customer classification dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def generate_sample_dataset(n_samples=1000, random_state=42):
    """Generates realistic customer classification dataset if CSV does not exist."""
    np.random.seed(random_state)
    genders = ['Male', 'Female']
    contracts = ['Month-to-month', 'One year', 'Two year']
    internets = ['DSL', 'Fiber optic', 'No']
    payments = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']

    data = []
    for i in range(1, n_samples + 1):
        cust_id = f"CUST-{i:04d}"
        gender = np.random.choice(genders)
        age = int(np.random.randint(18, 80))
        contract = np.random.choice(contracts, p=[0.55, 0.25, 0.20])
        tenure = int(np.random.randint(1, 25)) if contract == 'Month-to-month' else int(np.random.randint(12, 48)) if contract == 'One year' else int(np.random.randint(24, 73))
        internet = np.random.choice(internets, p=[0.35, 0.45, 0.20])
        num_products = int(np.random.randint(1, 7))
        base_charge = 20 + num_products * 8 + (45 if internet == 'Fiber optic' else 25 if internet == 'DSL' else 0)
        monthly_charges = round(base_charge + np.random.uniform(-7, 7), 2)
        total_charges = round(tenure * monthly_charges * np.random.uniform(0.95, 1.05), 2)
        payment = np.random.choice(payments, p=[0.35, 0.15, 0.25, 0.25])
        support_calls = int(np.clip(np.random.poisson(1.2 + (1.0 if contract == 'Month-to-month' else 0)), 0, 9))

        churn_score = 0.0
        if contract == 'Month-to-month': churn_score += 2.2
        elif contract == 'One year': churn_score -= 1.0
        else: churn_score -= 2.5
        churn_score -= min(3.0, (tenure / 12) * 0.7)
        if support_calls >= 4: churn_score += 2.8
        elif support_calls >= 2: churn_score += 1.2
        if internet == 'Fiber optic' and monthly_charges > 85: churn_score += 1.5
        if payment == 'Electronic check': churn_score += 0.8
        churn_score += np.random.normal(0, 0.7)
        customer_class = 1 if churn_score > 0.35 else 0

        data.append({
            'customer_id': cust_id, 'age': age, 'gender': gender, 'tenure': tenure,
            'monthly_charges': monthly_charges, 'total_charges': total_charges,
            'contract': contract, 'internet_service': internet, 'support_calls': support_calls,
            'payment_method': payment, 'num_products': num_products, 'customer_class': customer_class
        })
    return pd.DataFrame(data)

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    csv_path = os.path.join('data', 'customer_churn_data.csv')
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else generate_sample_dataset(1000)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)

    feature_cols = ['age', 'gender', 'tenure', 'monthly_charges', 'total_charges', 'contract', 'internet_service', 'support_calls', 'payment_method', 'num_products']
    X = df[feature_cols]
    y = df['customer_class']

    numeric_features = ['age', 'tenure', 'monthly_charges', 'total_charges', 'support_calls', 'num_products']
    categorical_features = ['gender', 'contract', 'internet_service', 'payment_method']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 1. Baseline Decision Tree
    dt_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))])
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt) * 100

    # 2. AdaBoost Classifier
    adaboost_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0, random_state=42))])
    adaboost_pipeline.fit(X_train, y_train)
    y_pred_ada = adaboost_pipeline.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred_ada) * 100

    print("=" * 55)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 55)
    print(f"Decision Tree Accuracy : {dt_acc:.2f}%")
    print(f"AdaBoost Accuracy      : {ada_acc:.2f}%")
    print(f"Accuracy Improvement   : +{ada_acc - dt_acc:.2f}%")
    print("=" * 55)

    joblib.dump(adaboost_pipeline, 'models/adaboost_pipeline.joblib')
    print("Saved trained model to models/adaboost_pipeline.joblib")

if __name__ == '__main__':
    main()
`;

export const PYTHON_APP_SCRIPT = `"""
Streamlit Web Application: Customer Classification using AdaBoost
College AI/ML Project
Run locally with: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Classification using AdaBoost", layout="wide")

st.title("Customer Classification using AdaBoost")
st.subheader("Interactive Machine Learning Prediction System")

model_path = os.path.join('models', 'adaboost_pipeline.joblib')
if not os.path.exists(model_path):
    st.warning("Trained model not found. Running training script...")
    import subprocess
    subprocess.run(["python", "train_model.py"], check=True)

pipeline = joblib.load(model_path)

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 90, 42)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (months)", 1, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c2:
        monthly = st.number_input("Monthly Charges ($)", 15.0, 150.0, 75.0)
        total = st.number_input("Total Charges ($)", 15.0, 9000.0, 900.0)
        support = st.slider("Support Calls (past 6 mo)", 0, 10, 2)
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        products = st.slider("Number of Products", 1, 6, 2)

    submit = st.form_submit_button("Predict Customer")

if submit:
    sample_df = pd.DataFrame([{
        'age': age, 'gender': gender, 'tenure': tenure, 'monthly_charges': monthly,
        'total_charges': total, 'contract': contract, 'internet_service': internet,
        'support_calls': support, 'payment_method': payment, 'num_products': products
    }])
    pred = pipeline.predict(sample_df)[0]
    proba = pipeline.predict_proba(sample_df)[0]

    if pred == 1:
        st.error(f"Customer Classification: **Churn Risk (At Risk)** with {proba[1]*100:.1f}% confidence")
    else:
        st.success(f"Customer Classification: **Retained (Loyal Customer)** with {proba[0]*100:.1f}% confidence")
`;

export const PYTHON_REQUIREMENTS = `pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
`;

export function triggerDownload(filename: string, content: string, mimeType = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
