# 🔮 ChurnGuard — Customer Churn Predictor

A Machine Learning web application that predicts whether a bank customer will **churn (leave)** based on their profile. Built with **Random Forest**, **Flask**, and a sleek dark-themed banking UI.

---

## 📸 Preview

<img width="1876" height="947" alt="Image" src="https://github.com/user-attachments/assets/3744493d-6c62-4974-995f-4e54ab1a810c" />
> Dark cyber-themed UI with real-time churn probability bar, risk badge, and model status indicator.

---

## 📁 Project Structure

```
churn-predictor/
│
├── Churn_Modelling.csv         # Raw dataset (10,000 customers)
├── churn_predictor.py          # Data cleaning, EDA, model training
├── app.py                      # Flask backend API
├── index.html                  # Frontend web UI
├── requirements.txt            # Python dependencies
│
│   ── Generated after training ──
├── ChurnModel.pkl              # Trained Random Forest model
├── scaler.pkl                  # Fitted StandardScaler
├── feature_names.pkl           # Feature column names
├── eda_plots.png               # Feature distribution plots
├── model_evaluation.png        # Confusion matrix + ROC curve
└── feature_importance.png      # Feature importance bar chart
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Churn_Modelling.csv (bank customer data) |
| Total rows | 10,000 |
| Target | `Exited` (0 = Stayed, 1 = Churned) |
| Churn rate | 20.4% |
| Null values | None |

### Features Used

| Feature | Type | Description |
|---|---|---|
| CreditScore | Numeric | Customer credit score (350–850) |
| Geography | Categorical | France / Germany / Spain |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years as customer (0–10) |
| Balance | Numeric | Account balance (€) |
| NumOfProducts | Numeric | Number of bank products (1–4) |
| HasCrCard | Binary | Has credit card (0/1) |
| IsActiveMember | Binary | Is active member (0/1) |
| EstimatedSalary | Numeric | Estimated annual salary (€) |

### Dropped Columns
- `RowNumber` — just an index
- `CustomerId` — unique identifier, no predictive value
- `Surname` — personal info, no predictive value

---

## 🤖 Machine Learning

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |
| max_depth | 10 |
| Train/Test Split | 80% / 20% (stratified) |
| Accuracy | **86.50%** |
| ROC-AUC | **0.8636** |

### Preprocessing Pipeline
1. Drop `RowNumber`, `CustomerId`, `Surname`
2. Label encode `Gender` (Female=0, Male=1)
3. One-hot encode `Geography` (drop_first=True)
4. `StandardScaler` applied to all features

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn RandomForestClassifier |
| Preprocessing | StandardScaler, LabelEncoder, pd.get_dummies |
| Visualization | matplotlib, seaborn |
| Backend API | Flask |
| Frontend | HTML5, CSS3, Vanilla JS |
| Model Persistence | pickle |

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
python churn_predictor.py
```

Expected output:
```
📁 Working directory: /your/project/
📄 Dataset shape: (10000, 14)
Churn rate: 20.4%
⏳ Training Random Forest Classifier...
✅ Training complete!
  Accuracy  : 86.50%
  ROC-AUC   : 0.8636
✅ Model saved   → ChurnModel.pkl
✅ Scaler saved  → scaler.pkl
✅ Features saved → feature_names.pkl
```

### 3. Start Flask server
```bash
python app.py
```
Server runs at `http://127.0.0.1:5000`

### 4. Open the web app
Open `index.html` in your browser. The status pill in the header turns **green** when Flask is connected.

---

## 🔗 How It Connects

```
Churn_Modelling.csv
       │
       ▼
churn_predictor.py ──trains──▶ ChurnModel.pkl + scaler.pkl
                                        │
index.html ──POST /predict──▶ app.py (Flask)
    ▲                               │
    └──── JSON { probability } ◀────┘
```

---

## 🌐 API Reference

### `POST /predict`

**Request:**
```json
{
  "credit_score": 650,
  "geography": "France",
  "gender": "Female",
  "age": 42,
  "tenure": 2,
  "balance": 80000,
  "num_products": 1,
  "has_cr_card": 1,
  "is_active": 1,
  "estimated_salary": 100000
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "probability": 67.3,
  "risk_level": "High",
  "label": "LIKELY TO CHURN"
}
```

### `GET /model-info`
```json
{
  "model": "Random Forest Classifier",
  "accuracy": "86.50%",
  "roc_auc": "0.8636",
  "trained_on": "10,000 customers",
  "features": ["CreditScore", "Gender", "Age", ...]
}
```

---

## 📈 Key Insights from the Data

- **Age** is the strongest churn predictor — older customers churn more
- **Germany** customers have higher churn rates than France/Spain
- **Inactive members** are significantly more likely to churn
- **Customers with 1 product** churn more than those with 2
- **High balance with low activity** is a strong churn signal

---

## ⚠️ Known Limitations

- Model trained only on European bank customer data — may not generalise globally
- Class imbalance (79.6% stay vs 20.4% churn) affects recall on the churn class
- Does not account for real-time behavioural data (transactions, complaints, etc.)

---

## 📄 License

MIT License — open source, free to use.
