# Housing Price Prediction Project

## Project Overview

This project aims to develop a reliable machine learning model for predicting housing prices. The model leverages various features of a housing property, such as area, number of rooms, and location, to estimate its market value. This solution can be valuable for real estate professionals, investors, and potential home buyers to make informed decisions in the housing market.

## Key Findings

*   **Model Performance:** XGBoostRegressor emerged as the best-performing model, achieving the highest R² score (0.7142), indicating a strong fit to the data. Gradient Boosting Regressor also demonstrated excellent performance with an R² score of 0.7086. Both models effectively capture the complex relationships between housing features and prices.

*   **Feature Importance:** Area proved to be the most significant predictor of housing prices, followed by the parking-to-stories ratio and bathroom count. Categorical features like furnishing status also had a noticeable impact.

*   **Impact of Feature Engineering and Data Transformation:** Feature engineering and log transformation of the target variable (housing prices) contributed to improved model performance.

## Project Deliverables

1.  **Trained Machine Learning Model:** The project successfully developed a high-performing XGBoostRegressor model for housing price prediction. This model is trained on a dataset of housing properties.
2.  **Jupyter Notebook:** A comprehensive Jupyter Notebook (`housing_price_prediction.ipynb`) documents the entire project workflow, including data exploration, feature engineering, model training, evaluation, and saving the trained model.
3.  **Training and Prediction Scripts:**
    *   `train.py`: This script trains the XGBoostRegressor model and saves it for future use.
    *   `predict.py`: This script allows users to make price predictions on new data using the saved model.

#### Project Structure
The project is organized into the following directories:
* config: Contains the configuration file (config.yaml) that stores project settings, such as file paths and hyperparameters.
* models: Stores the trained machine learning model (best_model.pkl) and other model-related files.
* scripts: Contains the training and prediction scripts (train.py and predict.py).

#### Configuration File
The config.yaml file stores project settings, such as file paths and hyperparameters. The file is used by the training and prediction scripts to load the correct configuration. The configuration file has the following structure:

```YAML
project:
  root_dir: "../"

paths:
  data_file: "data/Housing.csv"
  model_file: "models/best_model.pkl"

model:
  hyperparameters:
    learning_rate_range: [0.01, 0.5]
    n_estimators_range: [50, 300]
    max_depth_range: [3, 10]
    bayes_iterations: 10
```

## Running the Project Locally

### Data Source:
To download the Housing.csv file, visit
```
   https://www.kaggle.com/datasets/huyngohoang/housingcsv
```

### Notebook Name:
* housing_price_prediction.ipynb

### Prerequisites

To run this project, you will need the following installed on your machine:

*   Python 3.8 or higher
*   Jupyter Notebook
*   A virtual environment tool like `venv` or `conda` (recommended)
*   Required Python libraries (listed in `requirements.txt`)

### Setting Up the Environment

1. **Setting Up a Virtual Environment:**
Using venv:

```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS and Linux
    .\venv\Scripts\activate   # On Windows
```

Using conda:

```bash
    conda create -n housing-price-prediction python=3.8
    conda activate housing-price-prediction
```

2. **Clone the Repository:**
   
 ```bash
    git clone <repository-url>
    cd housing-price-prediction
```

3.  **Install Dependencies:**
  
```bash
    pip install -r requirements.txt
```
Packages:

```
    Package      Version
    ------------ -----------
    numpy==2.1.3
    pandas==2.2.3
    matplotlib==3.9.2
    seaborn==0.13.2
    scikit-learn==1.5.2
    lightgbm==4.5.0
    xgboost==2.1.2
    skopt==0.9.0
    flask==3.1.0
    yaml
```

## Instructions to Run the Project

### Running the Project

To run the project, follow these steps:
#### Step 1: Clone the Repository

Clone the repository using Git:

```bash
    git clone https://github.com/your-username/your-repo-name.git
```

Replace your-username and your-repo-name with your actual GitHub username and repository name.

#### Step 2: Install Dependencies:

Navigate to the project directory and install the dependencies using pip:

```bash
    cd your-repo-name
    pip install -r requirements.txt
```

#### Step 3: Train the Model:
Run the train.py script to train the model:

```bash
    python train.py
```

#### Expected output:

```
    Data Preparation
    Splitting dataset into features and target
    Train-Test Split
    Model Training and Evaluation
    Hyperparameter Tuning
    Save the best 
```

#### Step 4: Run the API
Run the predict_model.py script to start the API:

```bash
    python predict.py
```

#### Expected output:
* Serving Flask app 'predict_model'
* Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5000
Press CTRL+C to quit

#### Step 5: Make a Prediction
Use a tool like curl or Postman to send a POST request to http://127.0.0.1:5000/predict with a JSON payload containing the house details:

```bash: 

Windows:

Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes", "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}'

Linux:
curl -X POST -H "Content-Type: application/json" -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes", "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}' http://127.0.0.1:5000/predict

```

#### Expected Output
{"predicted_price": "$8,495,095.00"}

Note: The actual house price for this input is $13,300,000. The predicted price is approximately 36% lower than the actual price.
