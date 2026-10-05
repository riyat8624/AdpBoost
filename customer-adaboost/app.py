"""
Streamlit Web Application: Customer Classification using AdaBoost
College AI/ML Project
Run locally with: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Classification using AdaBoost",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained models
@st.cache_resource
def load_models():
    model_path = os.path.join('models', 'adaboost_pipeline.joblib')
    dt_path = os.path.join('models', 'dt_pipeline.joblib')
    
    if not os.path.exists(model_path):
        import subprocess
        st.info("Trained model not found. Running training script now...")
        subprocess.run(["python", "train_model.py"], check=True)
        
    ada_model = joblib.load(model_path)
    dt_model = joblib.load(dt_path) if os.path.exists(dt_path) else None
    return ada_model, dt_model

try:
    ada_pipeline, dt_pipeline = load_models()
except Exception as e:
    ada_pipeline, dt_pipeline = None, None
    st.error(f"Error loading model: {e}. Please ensure 'train_model.py' has been executed.")

# Sidebar Navigation
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Customer Prediction", "📊 Model Performance", "🧠 About AdaBoost"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Customer Classification using AdaBoost\n\n"
    "**Course:** AI / ML Project\n\n"
    "**Baseline:** Decision Tree (CART)\n\n"
    "**Ensemble:** AdaBoost.M1 (SAMME)"
)

# ----------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------
if page == "🏠 Home":
    st.title("Customer Classification using AdaBoost")
    st.subheader("Using Boosting Techniques to Improve Customer Classification Accuracy")
    
    st.markdown("""
    Welcome to the Machine Learning Customer Classification system. This project demonstrates how 
    **Adaptive Boosting (AdaBoost)** transforms a sequence of weak learners (Decision Stumps) into a high-accuracy, 
    robust strong classifier, outperforming a standard Decision Tree baseline.
    """)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🚀 Project Objectives")
        st.markdown("""
        * **Classify Customer Behavior**: Detect whether a customer is at **Churn Risk** or likely to remain **Retained**.
        * **Ensemble Learning Demonstration**: Compare baseline Decision Tree against sequential AdaBoost.
        * **Rigorous Performance Evaluation**: Track Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
        * **Real-Time Prediction Engine**: Input new customer records and receive immediate classification and confidence estimates.
        """)
        
    with col2:
        st.markdown("### 🔄 Core ML Workflow")
        st.markdown("""
        ```text
        [ Customer Dataset ] 
                 ↓
        [ Data Preprocessing (Imputation & One-Hot Encoding) ] 
                 ↓
        [ Stratified Train/Test Split (80/20) ] 
                 ↓
        ┌─────────────────────────┴─────────────────────────┐
        ↓                                                   ↓
        [ Baseline Decision Tree ]                 [ AdaBoost Classifier ]
        (Single CART Tree)                         (Sequential Weighted Stumps)
        └─────────────────────────┬─────────────────────────┘
                 ↓
        [ Performance Comparison & Confusion Matrix ]
                 ↓
        [ Interactive Customer Prediction System ]
        ```
        """)

# ----------------------------------------------------
# PAGE 2: CUSTOMER PREDICTION
# ----------------------------------------------------
elif page == "🎯 Customer Prediction":
    st.title("🎯 Customer Prediction")
    st.markdown("Enter customer details below to predict churn status using the trained AdaBoost model.")
    
    with st.expander("✨ Quick Preset Profiles", expanded=False):
        c1, c2, c3 = st.columns(3)
        if c1.button("Preset: High Churn Risk"):
            st.session_state['age'] = 52
            st.session_state['gender'] = 'Female'
            st.session_state['tenure'] = 3
            st.session_state['contract'] = 'Month-to-month'
            st.session_state['internet'] = 'Fiber optic'
            st.session_state['monthly'] = 94.5
            st.session_state['total'] = 283.5
            st.session_state['support'] = 5
            st.session_state['payment'] = 'Electronic check'
            st.session_state['products'] = 1
            st.rerun()
            
        if c2.button("Preset: Loyal Retained"):
            st.session_state['age'] = 41
            st.session_state['gender'] = 'Male'
            st.session_state['tenure'] = 48
            st.session_state['contract'] = 'Two year'
            st.session_state['internet'] = 'DSL'
            st.session_state['monthly'] = 58.0
            st.session_state['total'] = 2784.0
            st.session_state['support'] = 1
            st.session_state['payment'] = 'Bank transfer'
            st.session_state['products'] = 4
            st.rerun()
            
        if c3.button("Preset: Moderate / Boundary"):
            st.session_state['age'] = 34
            st.session_state['gender'] = 'Male'
            st.session_state['tenure'] = 14
            st.session_state['contract'] = 'One year'
            st.session_state['internet'] = 'Fiber optic'
            st.session_state['monthly'] = 75.0
            st.session_state['total'] = 1050.0
            st.session_state['support'] = 2
            st.session_state['payment'] = 'Credit card'
            st.session_state['products'] = 2
            st.rerun()

    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get('age', 40))
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get('gender', 'Male') == 'Male' else 1)
            tenure = st.number_input("Tenure (Months with company)", min_value=1, max_value=72, value=st.session_state.get('tenure', 24))
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0 if st.session_state.get('contract', 'Month-to-month') == 'Month-to-month' else 1 if st.session_state.get('contract') == 'One year' else 2)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1 if st.session_state.get('internet', 'Fiber optic') == 'Fiber optic' else 0 if st.session_state.get('internet') == 'DSL' else 2)
            
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=float(st.session_state.get('monthly', 70.0)))
            total_charges = st.number_input("Total Charges ($)", min_value=10.0, max_value=10000.0, value=float(st.session_state.get('total', float(tenure * monthly_charges))))
            support_calls = st.slider("Support Calls in Past 6 Months", min_value=0, max_value=10, value=int(st.session_state.get('support', 1)))
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], index=0 if st.session_state.get('payment', 'Electronic check') == 'Electronic check' else 2)
            num_products = st.slider("Number of Products/Services Subscribed", min_value=1, max_value=6, value=int(st.session_state.get('products', 2)))

        submitted = st.form_submit_button("⚡ Predict Customer", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract': contract,
            'internet_service': internet_service,
            'support_calls': support_calls,
            'payment_method': payment_method,
            'num_products': num_products
        }])

        if ada_pipeline is not None:
            pred = ada_pipeline.predict(input_df)[0]
            proba = ada_pipeline.predict_proba(input_df)[0]
            churn_prob = proba[1] * 100
            retained_prob = proba[0] * 100

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                if pred == 1:
                    st.error("### Customer Classification: ⚠️ Churn Risk")
                    st.metric("Confidence", f"{churn_prob:.1f}%")
                else:
                    st.success("### Customer Classification: ✅ Loyal / Retained")
                    st.metric("Confidence", f"{retained_prob:.1f}%")
                    
            with res_col2:
                st.markdown("#### Key Influencing Factors")
                if contract == 'Month-to-month':
                    st.warning("• Month-to-month contracts have high sensitivity to churn.")
                else:
                    st.info(f"• Long-term agreement ({contract}) promotes retention.")
                if support_calls >= 4:
                    st.error(f"• High frequency of support calls ({support_calls}) signals severe dissatisfaction.")
                if tenure >= 36:
                    st.success(f"• Long tenure ({tenure} months) reflects established brand trust.")

# ----------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# ----------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance & Comparison")
    
    csv_path = os.path.join('data', 'customer_churn_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.markdown(f"**Dataset Overview:** `{len(df)}` Samples | `10` Input Features | `80/20` Train/Test Split")
        
        # Load models to compute live test metrics
        X = df.drop(columns=['customer_id', 'customer_class', 'churn_status'], errors='ignore')
        y = df['customer_class']
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if ada_pipeline and dt_pipeline:
            y_pred_dt = dt_pipeline.predict(X_test)
            y_pred_ada = ada_pipeline.predict(X_test)
            
            dt_acc = accuracy_score(y_test, y_pred_dt) * 100
            ada_acc = accuracy_score(y_test, y_pred_ada) * 100
            diff = ada_acc - dt_acc
            
            st.success(f"### 🚀 AdaBoost Improvement = AdaBoost Accuracy ({ada_acc:.1f}%) − Decision Tree Accuracy ({dt_acc:.1f}%) = **+{diff:.1f}%**")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AdaBoost Accuracy", f"{ada_acc:.1f}%", f"+{diff:.1f}%")
            m2.metric("AdaBoost Precision", f"{precision_score(y_test, y_pred_ada)*100:.1f}%")
            m3.metric("AdaBoost Recall", f"{recall_score(y_test, y_pred_ada)*100:.1f}%")
            m4.metric("AdaBoost F1-Score", f"{f1_score(y_test, y_pred_ada)*100:.1f}%")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Model Comparison Bar Chart")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
                    'Score (%)': [
                        dt_acc, precision_score(y_test, y_pred_dt)*100, recall_score(y_test, y_pred_dt)*100, f1_score(y_test, y_pred_dt)*100,
                        ada_acc, precision_score(y_test, y_pred_ada)*100, recall_score(y_test, y_pred_ada)*100, f1_score(y_test, y_pred_ada)*100
                    ],
                    'Model': ['Decision Tree (Baseline)'] * 4 + ['AdaBoost (Boosted)'] * 4
                })
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=metrics_df, x='Metric', y='Score (%)', hue='Model', palette=['#94a3b8', '#38bdf8'], ax=ax)
                ax.set_ylim(0, 100)
                st.pyplot(fig)
                
            with col_chart2:
                st.subheader("AdaBoost Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_ada)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Retained', 'Churn'], yticklabels=['Retained', 'Churn'], ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
    else:
        st.warning("Dataset not found. Run train_model.py first.")

# ----------------------------------------------------
# PAGE 4: ABOUT ADABOOST
# ----------------------------------------------------
elif page == "🧠 About AdaBoost":
    st.title("🧠 Understanding AdaBoost")
    st.markdown("""
    ### Why AdaBoost?
    A single Decision Tree or Decision Stump is often a **weak learner** with modest accuracy.
    AdaBoost (Adaptive Boosting) combines multiple weak learners into a powerful strong classifier.
    
    ### Sequential Boosting Mechanism
    Unlike Bagging (e.g. Random Forest) which trains trees independently in parallel, 
    **AdaBoost learns sequentially**:
    1. **Initial Uniform Weights**: Every sample starts with weight $w_i = \\frac{1}{N}$.
    2. **Weak Learner Fitting**: A decision stump $h_t(x)$ is trained minimizing weighted error $\\epsilon_t$.
    3. **Estimator Importance ($\\alpha_t$)**: Estimators with low error receive higher voting weight:
       $$\\alpha_t = \\frac{1}{2} \\ln\\left(\\frac{1 - \\epsilon_t}{\\epsilon_t}\\right)$$
    4. **Sample Re-weighting**: Misclassified instances have their weights increased, forcing subsequent learners to prioritize hard cases.
    5. **Final Voting**: The ensemble aggregates weighted votes:
       $$H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)\\right)$$
    """)
    st.code("Weak Learner 1 → Reweight Misclassified → Weak Learner 2 → Reweight → Weak Learner 3 → Weighted Combination → Final Strong Classifier", language="text")
