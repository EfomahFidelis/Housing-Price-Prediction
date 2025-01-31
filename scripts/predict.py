from flask import Flask, request, jsonify
import pickle
import os
import yaml
import numpy as np

def load_config(filepath='config/config.yaml'):
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        config_file = os.path.join(parent_dir, filepath)
        
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("Configuration file not found.")
        exit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit()

config = load_config()
model_file = os.path.join(config['project']['root_dir'], config['paths']['model_file'])

# Load model and DictVectorizer
try:
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
except FileNotFoundError:
    print("Model file not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit()

# Initialize Flask app
app = Flask(__name__)

def validate_input(data):
    required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    for field in required_fields:
        if field not in data:
            return False
        if not isinstance(data[field], (int, float, str, bool)):
            return False
    return True

@app.route('/predict', methods=['POST'])
def predict():
    try:
        house = request.get_json()

        if not validate_input(house):
            return jsonify({'error': 'Invalid input data'}), 400

        house_dict = [house]
        X = dv.transform(house_dict)
        predicted_price = model.predict(X)[0]

        return jsonify({'predicted_price': predicted_price})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)