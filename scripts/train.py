import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import pickle


def get_config_value(config, *keys, default=None):
    """Safely access nested config values."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def load_config():
    filepath = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    else:
        raise FileNotFoundError("Configuration file not found. Please ensure 'config/config.yaml' exists.")

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def train(X_train, y_train):
    numerical = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    categorical = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                  'airconditioning', 'prefarea']

    df_train = X_train.copy()

    for col in categorical + ['furnishingstatus']:
        df_train[col] = df_train[col].str.lower()

    furnishing_dummies = pd.get_dummies(
        df_train['furnishingstatus'], 
        prefix='furnishing',
        drop_first=True
    )

    df_train = df_train.drop('furnishingstatus', axis=1)
    df_train = pd.concat([df_train, furnishing_dummies], axis=1)

    for col in categorical:
        df_train[col] = (df_train[col] == 'yes').astype(int)

    feature_columns = numerical + categorical + list(furnishing_dummies.columns)
    df_train = df_train[feature_columns]

    dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train_transformed = dv.fit_transform(dicts)

    model = XGBRegressor(random_state=42)
    model.fit(X_train_transformed, y_train)

    return dv, model

def evaluate_model(model, dv, X_test, y_test):
    dicts_test = X_test.to_dict(orient='records')
    X_test_transformed = dv.transform(dicts_test)

    y_pred = model.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

def tune_hyperparameters(X_train, y_train, config):
    model = XGBRegressor(random_state=42)
    search_space = {
        "learning_rate": tuple(config["model"]["hyperparameters"]["learning_rate_range"]),
        "n_estimators": tuple(config["model"]["hyperparameters"]["n_estimators_range"]),
        "max_depth": tuple(config["model"]["hyperparameters"]["max_depth_range"])
    }
    bayes_search = BayesSearchCV(model, search_space, n_iter=config["model"]["hyperparameters"]["bayes_iterations"], cv=3, random_state=42)
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_estimator_

def save_model(dv, model, filepath):
    with open(filepath, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

def main(): 
    # Load the configuration
    config = load_config()

    # Use the safe configuration value function
    filepath = os.path.join(
        get_config_value(config, 'project', 'root_dir', default='.'),
        get_config_value(config, 'paths', 'data_file', default='Housing.csv')
    )

    print("Data Preparation")
    df = load_data(filepath)
    X, y = preprocess_data(df)

    print("Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Model Training")
    dv, model = train(X_train, y_train)

    print("Model Evaluation")
    evaluate_model(model, dv, X_test, y_test)

    print("Hyperparameter Tuning")
    dicts_train = X_train.to_dict(orient='records')
    X_train_transformed = dv.transform(dicts_train)

    tuned_model = tune_hyperparameters(X_train_transformed, y_train, config)

    print("Saving model")
    model_filepath = os.path.join(config['project']['root_dir'], config['paths']['model_file'])
    save_model(dv, tuned_model, model_filepath)

if __name__ == '__main__':
    main()
