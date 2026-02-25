from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model         = pickle.load(open(os.path.join(BASE_DIR, 'ChurnModel.pkl'),    'rb'))
scaler        = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'),        'rb'))
feature_names = pickle.load(open(os.path.join(BASE_DIR, 'feature_names.pkl'), 'rb'))

print("✅ Model loaded | Features:", feature_names)
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()

        credit_score  = int(d['credit_score'])
        geography     = str(d['geography'])        
        gender        = str(d['gender'])            
        age           = int(d['age'])
        tenure        = int(d['tenure'])
        balance       = float(d['balance'])
        num_products  = int(d['num_products'])
        has_cr_card   = int(d['has_cr_card'])      
        is_active     = int(d['is_active'])        
        salary        = float(d['estimated_salary'])

        gender_enc   = 1 if gender == 'Male' else 0
        geo_germany  = 1 if geography == 'Germany' else 0
        geo_spain    = 1 if geography == 'Spain'   else 0

        row = pd.DataFrame([[credit_score, gender_enc, age, tenure, balance,
                              num_products, has_cr_card, is_active, salary,
                              geo_germany, geo_spain]], columns=feature_names)

        row_sc = scaler.transform(row)
        pred   = int(model.predict(row_sc)[0])
        prob   = float(model.predict_proba(row_sc)[0][1])

        if prob < 0.30:   risk = 'Low'
        elif prob < 0.60: risk = 'Medium'
        else:             risk = 'High'

        return jsonify({
            'success':    True,
            'prediction': pred,           
            'probability': round(prob * 100, 1),
            'risk_level': risk,
            'label': 'LIKELY TO CHURN' if pred == 1 else 'LIKELY TO STAY'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    return jsonify({
        'model': 'Random Forest Classifier',
        'accuracy': '86.50%',
        'roc_auc': '0.8636',
        'trained_on': '10,000 customers',
        'features': feature_names
    })

if __name__ == '__main__':
    print("\n🚀 Churn Predictor server starting...")
    print("   Open http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
