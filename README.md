# Housing-Price-

# Housing Price Prediction Project

## Project Overview

This project aims to develop a reliable machine learning model for predicting housing prices in Ikorodu, Lagos, Nigeria. The model leverages various features of a housing property, such as area, number of rooms, and location, to estimate its market value. This solution can be valuable for real estate professionals, investors, and potential home buyers to make informed decisions in the housing market.

## Key Findings

* **Model Performance:** XGBoostRegressor emerged as the best performing model, achieving the highest R² score (0.7142), indicating a strong fit to the data. Gradient Boosting Regressor also demonstrated excellent performance with an R² score of 0.7086. Both models effectively capture the complex relationships between housing features and prices.
* **Feature Importance:** Area proved to be the most significant predictor of housing prices, followed by parking-to-stories ratio and bathroom count. Categorical features like furnishing status also had a noticeable impact.
* **Impact of Feature Engineering and Data Transformation:** Feature engineering and log transformation of the target variable (housing prices) contributed to improved model performance.

## Project Deliverables

1. **Trained Machine Learning Model:** The project successfully developed a high-performing XGBoostRegressor model for housing price prediction. This model is trained on a dataset of housing properties in Ikorodu, Lagos, Nigeria.
2. **Jupyter Notebook:** A comprehensive Jupyter Notebook (`notebook.ipynb`) documents the entire project workflow, including data exploration, feature engineering, model training, evaluation, and saving the trained model.
3. **Training and Prediction Scripts:** 
    * `train.py`: This script trains the XGBoostRegressor model and saves it for future use.
    * `predict.py`: This script allows users to make price predictions on new data using the saved model.

## Running the Project Locally

### Prerequisites

To run this project, you'll need the following installed on your machine:

* Python 3.8 or higher
* Jupyter Notebook
* A virtual environment tool like `venv` or `conda` (recommended)
* Required Python libraries (listed in `requirements.txt`)

### Setting Up the Environment

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd housing-price-prediction