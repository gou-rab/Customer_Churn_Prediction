import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Working directory: {BASE_DIR}")

CSV_PATH = os.path.join(BASE_DIR, 'Churn_Modelling.csv')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"\n📄 Dataset shape: {df.shape}")
print(df.head(3))
print("\nTarget distribution:")
print(df['Exited'].value_counts())
print(f"\nChurn rate: {df['Exited'].mean()*100:.1f}%")

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])   

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

print(f"\n✅ Features after encoding: {list(df.columns)}")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0d1117')

plot_configs = [
    ('Age',             'Age Distribution by Churn',   'age'),
    ('Balance',         'Balance Distribution',         'balance'),
    ('CreditScore',     'Credit Score Distribution',   'credit'),
    ('EstimatedSalary', 'Estimated Salary',            'salary'),
    ('Tenure',          'Tenure (Years)',               'tenure'),
    ('NumOfProducts',   'Number of Products',          'products'),
]

colors = ['#00f5ff', '#ff4757']

for ax, (col, title, _) in zip(axes.flat, plot_configs):
    ax.set_facecolor('#161b22')
    for exited, color in zip([0, 1], colors):
        data = df[df['Exited'] == exited][col]
        ax.hist(data, bins=25, alpha=0.7, color=color,
                label='Stayed' if exited == 0 else 'Churned', density=True)
    ax.set_title(title, color='white', fontsize=11, pad=8)
    ax.legend(fontsize=8)
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

plt.suptitle('Customer Churn — Feature Distributions', color='white',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'eda_plots.png'), dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("📊 EDA plots saved.")

X = df.drop('Exited', axis=1)
y = df['Exited']

feature_names = list(X.columns)
print(f"\n🔢 Feature count: {len(feature_names)}")
print(f"   Features: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {X_train.shape} | Test: {X_test.shape}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n⏳ Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)
print("✅ Training complete!")

y_pred      = model.predict(X_test_sc)
y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'='*45}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"{'='*45}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#161b22')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'],
            linewidths=0.5)
ax.set_title('Confusion Matrix', color='white', fontsize=12)
ax.set_ylabel('Actual', color='white')
ax.set_xlabel('Predicted', color='white')
ax.tick_params(colors='white')

ax2 = axes[1]
ax2.set_facecolor('#161b22')
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax2.plot(fpr, tpr, color='#00f5ff', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
ax2.plot([0,1],[0,1], color='#555', linestyle='--', lw=1)
ax2.fill_between(fpr, tpr, alpha=0.1, color='#00f5ff')
ax2.set_title('ROC Curve', color='white', fontsize=12)
ax2.set_xlabel('False Positive Rate', color='white')
ax2.set_ylabel('True Positive Rate', color='white')
ax2.legend(fontsize=10)
ax2.tick_params(colors='#888')
for spine in ax2.spines.values():
    spine.set_edgecolor('#30363d')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'model_evaluation.png'), dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("📈 Model evaluation plot saved.")

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
               color='#00f5ff', edgecolor='none', height=0.6)
ax.set_title('Feature Importance (Random Forest)', color='white', fontsize=13)
ax.set_xlabel('Importance Score', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'feature_importance.png'), dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("📊 Feature importance plot saved.")

pickle.dump(model,        open(os.path.join(BASE_DIR, 'ChurnModel.pkl'),    'wb'))
pickle.dump(scaler,       open(os.path.join(BASE_DIR, 'scaler.pkl'),        'wb'))
pickle.dump(feature_names,open(os.path.join(BASE_DIR, 'feature_names.pkl'),'wb'))

print(f"\n✅ Model saved   → ChurnModel.pkl")
print(f"✅ Scaler saved  → scaler.pkl")
print(f"✅ Features saved → feature_names.pkl")

def predict_churn(credit_score, geography, gender, age, tenure,
                  balance, num_products, has_cr_card, is_active, salary):
    m = pickle.load(open(os.path.join(BASE_DIR, 'ChurnModel.pkl'),    'rb'))
    s = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'),        'rb'))
    f = pickle.load(open(os.path.join(BASE_DIR, 'feature_names.pkl'), 'rb'))

    gender_enc = 1 if gender.lower() == 'male' else 0
    geo_germany = 1 if geography == 'Germany' else 0
    geo_spain   = 1 if geography == 'Spain'   else 0

    row = pd.DataFrame([[credit_score, gender_enc, age, tenure, balance,
                         num_products, has_cr_card, is_active, salary,
                         geo_germany, geo_spain]], columns=f)
    row_sc = s.transform(row)
    prob   = m.predict_proba(row_sc)[0][1]
    pred   = m.predict(row_sc)[0]
    return pred, round(prob * 100, 1)


print("\n" + "="*50)
print("🔮  DEMO PREDICTIONS")
print("="*50)
demos = [
    (600, 'France',  'Female', 42, 2,  0,      1, 1, 1, 101348, "High-risk profile"),
    (750, 'Germany', 'Male',   35, 8,  120000, 2, 1, 0, 85000,  "Medium-risk"),
    (820, 'Spain',   'Male',   28, 5,  50000,  1, 0, 1, 150000, "Low-risk"),
]
for cs, geo, gen, age, ten, bal, np_, hcc, iam, sal, label in demos:
    pred, prob = predict_churn(cs, geo, gen, age, ten, bal, np_, hcc, iam, sal)
    status = "⚠️  CHURN" if pred == 1 else "✅ STAY"
    print(f"  {label:25s} → {status}  ({prob}% churn probability)")
