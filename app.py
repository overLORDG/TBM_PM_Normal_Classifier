from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained classifiers and scaler
with open('TBM_classifier.pkl', 'rb') as file:
    tbm_classifier = pickle.load(file)

with open('PM_classifier.pkl', 'rb') as file:
    pm_classifier = pickle.load(file)

with open('Normal_classifier.pkl', 'rb') as file:
    normal_classifier = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the rule-based classification logic
def rule_based_classification(tlc, lymphocytes, polymorphs, protein, sugar):
    if tlc < 5 and 0 <= lymphocytes <= 10 and 0 <= polymorphs <= 10 and 6 <= protein <= 8 and 70 <= sugar <= 99:
        return "Normal"
    elif 5 <= tlc <= 250 and 70 <= lymphocytes <= 100 and 0 <= polymorphs <= 30 and 58 <= protein <= 200 and 20 <= sugar <= 50:
        return "TBM"
    elif tlc > 250 and 0 <= lymphocytes <= 20 and 20 <= polymorphs <= 100 and protein > 200 and 2 <= sugar <= 30:
        return "PM"
    return None

# Define a prediction function
def predict_class(input_data):
    features_scaled = scaler.transform([input_data])

    # Predict probabilities using each classifier
    probs = {
        'TBM': tbm_classifier.predict_proba(features_scaled)[:, 1][0] * 100,
        'PM': pm_classifier.predict_proba(features_scaled)[:, 1][0] * 100,
        'Normal': normal_classifier.predict_proba(features_scaled)[:, 1][0] * 100
    }

    # Get the ML-predicted class
    ml_predicted_class = max(probs, key=probs.get)

    # Rule-based prediction
    rule_based_prediction = rule_based_classification(*input_data)
    
    # Final class is determined by rule-based logic if applicable
    final_class = rule_based_prediction if rule_based_prediction else ml_predicted_class

    return final_class, probs

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_values = [
            float(request.form['TLC']),
            float(request.form['Lymphocytes']),
            float(request.form['Polymorphs']),
            float(request.form['Protein']),
            float(request.form['Sugar'])
        ]

        # Perform prediction
        predicted_class, probabilities = predict_class(input_values)

        # Prepare the response
        response = {
            'Predicted Class': predicted_class,
            'TBM Probability (%)': round(probabilities['TBM'], 2),
            'PM Probability (%)': round(probabilities['PM'], 2),
            'Normal Probability (%)': round(probabilities['Normal'], 2)
        }

        return render_template('result.html', result=response)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=False)
