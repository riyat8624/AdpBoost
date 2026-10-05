"""
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
        
        contract_prob = [0.55, 0.25, 0.20]
        contract = np.random.choice(contracts, p=contract_prob)
        
        if contract == 'Month-to-month':
            tenure = int(np.random.randint(1, 25))
        elif contract == 'One year':
            tenure = int(np.random.randint(12, 48))
        else:
            tenure = int(np.random.randint(24, 73))
            
        internet = np.random.choice(internets, p=[0.35, 0.45, 0.20])
        num_products = int(np.random.randint(1, 7))
        
        base_charge = 20 + num_products * 8
        if internet == 'Fiber optic':
            base_charge += 45
        elif internet == 'DSL':
            base_charge += 25
        monthly_charges = round(base_charge + np.random.uniform(-7, 7), 2)
        total_charges = round(tenure * monthly_charges * np.random.uniform(0.95, 1.05), 2)
        
        payment = np.random.choice(payments, p=[0.35, 0.15, 0.25, 0.25])
        
        lambda_support = 1.2
        if contract == 'Month-to-month':
            lambda_support += 1.0
        if internet == 'Fiber optic':
            lambda_support += 0.8
        support_calls = int(np.clip(np.random.poisson(lambda_support), 0, 9))
        
        # Ground truth non-linear churn indicator
        churn_score = 0.0
        if contract == 'Month-to-month':
            churn_score += 2.2
        elif contract == 'One year':
            churn_score -= 1.0
        else:
            churn_score -= 2.5
            
        churn_score -= min(3.0, (tenure / 12) * 0.7)
        if support_calls >= 4:
            churn_score += 2.8
        elif support_calls >= 2:
            churn_score += 1.2
        else:
            churn_score -= 0.6
            
        if internet == 'Fiber optic' and monthly_charges > 85:
            churn_score += 1.5
        if internet == 'No':
            churn_score -= 1.2
            
        if payment == 'Electronic check':
            churn_score += 0.8
        elif payment in ['Bank transfer', 'Credit card']:
            churn_score -= 0.7
            
        if age > 60:
            churn_score += 0.4
        if num_products >= 4:
            churn_score -= 0.9
            
        churn_score += np.random.normal(0, 0.7)
        customer_class = 1 if churn_score > 0.35 else 0

        data.append({
            'customer_id': cust_id,
            'age': age,
            'gender': gender,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract': contract,
            'internet_service': internet,
            'support_calls': support_calls,
            'payment_method': payment,
            'num_products': num_products,
            'customer_class': customer_class
        })

    return pd.DataFrame(data)

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    csv_path = os.path.join('data', 'customer_churn_data.csv')
    if os.path.exists(csv_path):
        print(f"[1/5] Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("[1/5] Generating benchmark customer dataset (1000 samples)...")
        df = generate_sample_dataset(1000, random_state=42)
        df.to_csv(csv_path, index=False)
        print(f"      Saved dataset to {csv_path}")

    # Features and Target
    feature_cols = [
        'age', 'gender', 'tenure', 'monthly_charges', 'total_charges',
        'contract', 'internet_service', 'support_calls', 'payment_method', 'num_products'
    ]
    target_col = 'customer_class'

    X = df[feature_cols]
    y = df[target_col]

    # Preprocessing pipelines
    numeric_features = ['age', 'tenure', 'monthly_charges', 'total_charges', 'support_calls', 'num_products']
    categorical_features = ['gender', 'contract', 'internet_service', 'payment_method']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio := 0.20, random_state=42, stratify=y
    )
    print(f"[2/5] Stratified Train/Test Split: {len(X_train)} Train samples, {len(X_test)} Test samples")

    # 1. Baseline Model: Decision Tree Classifier
    print("[3/5] Training Baseline Model: Decision Tree Classifier (max_depth=3)...")
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)

    dt_acc = accuracy_score(y_test, y_pred_dt) * 100
    dt_prec = precision_score(y_test, y_pred_dt) * 100
    dt_rec = recall_score(y_test, y_pred_dt) * 100
    dt_f1 = f1_score(y_test, y_pred_dt) * 100

    # 2. Boosting Model: AdaBoost Classifier
    print("[4/5] Training Boosted Model: AdaBoost Classifier (n_estimators=50, learning_rate=1.0)...")
    adaboost_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ))
    ])
    adaboost_pipeline.fit(X_train, y_train)
    y_pred_ada = adaboost_pipeline.predict(X_test)

    ada_acc = accuracy_score(y_test, y_pred_ada) * 100
    ada_prec = precision_score(y_test, y_pred_ada) * 100
    ada_rec = recall_score(y_test, y_pred_ada) * 100
    ada_f1 = f1_score(y_test, y_pred_ada) * 100
    cm_ada = confusion_matrix(y_test, y_pred_ada)

    # 3. Model Comparison
    acc_diff = ada_acc - dt_acc
    print("\n" + "=" * 60)
    print("           MODEL PERFORMANCE COMPARISON (TEST DATASET)")
    print("=" * 60)
    print(f"{'Metric':<22} | {'Decision Tree (Baseline)':<24} | {'AdaBoost (Boosted)':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<22} | {dt_acc:>23.2f}% | {ada_acc:>19.2f}%")
    print(f"{'Precision':<22} | {dt_prec:>23.2f}% | {ada_prec:>19.2f}%")
    print(f"{'Recall':<22} | {dt_rec:>23.2f}% | {ada_rec:>19.2f}%")
    print(f"{'F1 Score':<22} | {dt_f1:>23.2f}% | {ada_f1:>19.2f}%")
    print("-" * 60)
    print(f"AdaBoost Accuracy Improvement: +{acc_diff:.2f}%")
    print("=" * 60)

    print("\nAdaBoost Confusion Matrix:")
    print(f"[[TN={cm_ada[0,0]:<3}  FP={cm_ada[0,1]:<3}]")
    print(f" [FN={cm_ada[1,0]:<3}  TP={cm_ada[1,1]:<3}]]")

    # 4. Save Trained AdaBoost Pipeline
    model_save_path = os.path.join('models', 'adaboost_pipeline.joblib')
    dt_save_path = os.path.join('models', 'dt_pipeline.joblib')
    joblib.dump(adaboost_pipeline, model_save_path)
    joblib.dump(dt_pipeline, dt_save_path)
    print(f"\n[5/5] Trained AdaBoost model saved to {model_save_path}")
    print("      Model is ready for fast inference in web app!")

if __name__ == '__main__':
    main()
