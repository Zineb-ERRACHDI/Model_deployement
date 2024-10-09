from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form.to_dict()
    
    # Convert input to float and prepare for prediction
    new_data = [[float(data['MedInc']), float(data['HouseAge']), float(data['AveRooms']),
                  float(data['AveBedrms']), float(data['Population']), float(data['AveOccup']),
                  float(data['Latitude']), float(data['Longitude'])]]
    
    # Scale the input data
    new_data_scaled = loaded_scaler.transform(new_data)
    
    # Make a prediction
    prediction = loaded_model.predict(new_data_scaled)
    rounded_prediction = round(prediction[0], 4)
    
    # Return the result to the user
    return render_template('index.html', prediction=rounded_prediction)

if __name__ == '__main__':
    app.run(debug=True)
