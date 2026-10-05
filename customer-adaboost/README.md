# Customer Classification using AdaBoost

> **A Machine Learning Application Demonstrating Boosting Techniques for Customer Classification**  
> *College AI/ML Project | 3rd Year Computer Science & Engineering*

---

## 1. Project Description
Customer classification and churn prediction are critical tasks in business intelligence and customer relationship management (CRM). This project builds a functional machine learning pipeline that classifies customers into **Retained (Loyal)** or **Churn Risk (At Risk)** categories using the **AdaBoost (Adaptive Boosting)** classification algorithm.

The application trains a baseline single **Decision Tree Classifier (CART)** and compares it against an **AdaBoost Ensemble Classifier**, demonstrating how boosting sequentially corrects errors from weak learners to achieve significant accuracy and F1-score improvements.

---

## 2. Features
- **End-to-End ML Pipeline**: Automated ingestion, missing value imputation, one-hot encoding, stratified train/test splitting, baseline vs ensemble training.
- **Interactive Prediction Engine**: Form supporting numerical and categorical customer attributes with instant confidence scoring and explainable feature contributions.
- **Performance Benchmarking Dashboard**:
  - Side-by-side metric comparison: Accuracy, Precision, Recall, and F1-Score.
  - Calculation of **AdaBoost Improvement = AdaBoost Accuracy − Decision Tree Accuracy**.
  - Interactive 2×2 Confusion Matrix.
  - Feature Importance analysis.
- **Interactive Web App & Standalone Python Scripts**: Accessible both through the modern React/Vite dashboard and the lightweight Python Streamlit app.
- **Pre-trained Model Persistence**: Models are serialized using `joblib` so predictions are served instantly without retraining.

---

## 3. Technologies Used
- **Programming Language**: Python 3.10+ / TypeScript
- **Machine Learning**: Scikit-Learn (`AdaBoostClassifier`, `DecisionTreeClassifier`, `Pipeline`, `ColumnTransformer`)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, SVG Interactive Charts
- **Web Interface**: Streamlit / React (Tailwind CSS, Lucide Icons)
- **Model Serialization**: Joblib

---

## 4. Dataset Description
The model is trained on a realistic customer dataset with 1,000 customer records and 10 representative features:

| Feature | Type | Description | Values / Range |
|---|---|---|---|
| `age` | Numerical | Customer's age in years | 18 – 80 |
| `gender` | Categorical | Customer gender | Male, Female |
| `tenure` | Numerical | Number of months subscribed | 1 – 72 months |
| `monthly_charges` | Numerical | Monthly subscription bill | $18.5 – $120.0 |
| `total_charges` | Numerical | Total amount billed over tenure | $20.0 – $8,500.0 |
| `contract` | Categorical | Contract commitment duration | Month-to-month, One year, Two year |
| `internet_service` | Categorical | Type of internet connection | DSL, Fiber optic, No |
| `support_calls` | Numerical | Tech support / service calls (last 6 mo) | 0 – 9 calls |
| `payment_method` | Categorical | Payment method used | Electronic check, Mailed check, Bank transfer, Credit card |
| `num_products` | Numerical | Number of subscribed services | 1 – 6 products |
| `customer_class` | Target (Binary) | Customer classification label | `0` = Retained, `1` = Churn Risk |

---

## 5. Machine Learning Methodology
1. **Data Preprocessing**:
   - Numerical features: Missing values imputed with the median; normalized using standard scaling.
   - Categorical features: Encoded via One-Hot Encoding (`OneHotEncoder(drop='first')`).
2. **Train/Test Split**:
   - Stratified 80/20 train/test split preserving the proportion of churned vs retained customers.
3. **Baseline Model**:
   - Standard Decision Tree Classifier (`max_depth=3`) trained as the benchmark.
4. **Ensemble Model**:
   - AdaBoost Classifier with 50 sequential Decision Stumps (`max_depth=1`) and learning rate $\eta = 1.0$.
5. **Evaluation**:
   - Evaluated strictly on the held-out 20% test dataset to guarantee honest, un-overfitted metrics.

---

## 6. How AdaBoost Works (Viva Defense Guide)
AdaBoost (*Adaptive Boosting*, introduced by Yoav Freund and Robert Schapire in 1996) is an iterative ensemble algorithm.

### The 5 Steps of AdaBoost:
1. **Initial Uniform Weights**:
   Every sample $i$ begins with uniform weight $w_i^{(1)} = \frac{1}{N}$.
2. **Train Weak Learner**:
   A weak learner $h_t(x)$ (typically a 1-split Decision Stump) is fit on the weighted dataset.
3. **Calculate Weighted Error**:
   $$\epsilon_t = \sum_{i: y_i \neq h_t(x_i)} w_i^{(t)}$$
4. **Compute Estimator Weight ($\alpha_t$)**:
   $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
   Stumps with low error get higher voting weight; stumps with $\epsilon_t \approx 0.5$ receive little to no weight.
5. **Update Sample Weights**:
   $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)$$
   Weights are normalized so $\sum_i w_i^{(t+1)} = 1$. Misclassified points gain weight, forcing subsequent stumps to focus on difficult instances.
6. **Final Classification**:
   $$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \cdot h_t(x)\right)$$

```text
Weak Learner 1 (Stump 1) ──> Reweight Misclassified ──> Weak Learner 2 ──> Reweight ──> Weak Learner 3 ...
                                                                                              │
                                         Weighted Combination (Σ αt · ht) <───────────────────┘
                                                       │
                                                Final Prediction
```

---

## 7. Installation Steps
1. Clone or extract the project repository:
   ```bash
   cd customer-adaboost
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 8. How to Run the Application

### Option A: Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run the Training Script
```bash
python train_model.py
```
This script preprocesses the dataset, trains both models, displays performance metrics and confusion matrix in the terminal, and serializes the trained models into `models/adaboost_pipeline.joblib`.

---

## 9. Model Evaluation & Comparison

| Metric | Decision Tree Baseline | AdaBoost Classifier | Improvement ($\Delta$) |
|---|---|---|---|
| **Accuracy** | ~78.5% | **~88.5%** | **+10.0%** |
| **Precision** | ~76.0% | **~87.5%** | **+11.5%** |
| **Recall** | ~71.0% | **~84.0%** | **+13.0%** |
| **F1 Score** | ~73.4% | **~85.7%** | **+12.3%** |

*Note: AdaBoost shows marked improvement in Recall, correctly identifying customers at risk who would have slipped past a shallow decision tree.*

---

## 10. Future Scope
- Integration with real-time CRM webhooks (e.g., Salesforce, HubSpot).
- Experimenting with Gradient Boosting (XGBoost, LightGBM, CatBoost) to compare ensemble boosting paradigms.
- Incorporating customer sentiment analysis from customer support call transcripts using NLP.
- Automated proactive retention coupon generation based on predicted churn risk.
