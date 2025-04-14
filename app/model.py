import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import json


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model


def get_model_metrics(model_path, training_data):
    """
    Retrieve model performance metrics.

    Args:
        model_path (str): Path to model.
        training_data (pd.DataFrame): Training data for num_countries.

    Returns:
        dict: {'rmse': float, 'training_date': str, 'num_countries': int}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mtime = os.path.getmtime(model_path)
    training_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        rmse = metrics.get('rmse', 0.0)

    num_countries = len(
        training_data['country'].unique()) if not training_data.empty else 0

    return {
        'rmse': rmse,
        'training_date': training_date,
        'num_countries': num_countries
    }


def predict_country(country, date, model_path, training_data):
    """
    Predict revenue for a country on a given date.

    Args:
        country (str): Country name.
        date (datetime): Prediction date.
        model_path (str): Path to model.
        training_data (pd.DataFrame): Training data for encoder and lag.

    Returns:
        float: Predicted revenue.
    """
    # Load model
    model = load_model(model_path)

    # Validate country
    if country not in training_data['country'].unique():
        raise ValueError(f"Country not found: {country}")

    # Prepare features
    label_encoder = LabelEncoder()
    label_encoder.fit(training_data['country'])
    country_encoded = label_encoder.transform([country])[0]

    # Get latest revenue_lag1 for country
    country_data = training_data[training_data['country']
                                 == country].sort_values('date')
    revenue_lag1 = country_data['revenue'].iloc[-1] if not country_data.empty else 0.0

    # Extract date features
    year = date.year
    month = date.month

    # Assume times_viewed=0 (no input provided)
    times_viewed = 0.0

    # Create input DataFrame
    input_data = pd.DataFrame({
        'country_encoded': [country_encoded],
        'month': [month],
        'year': [year],
        'times_viewed': [times_viewed],
        'revenue_lag1': [revenue_lag1]
    })

    # Predict
    revenue = model.predict(input_data)[0]
    return revenue


def predict_all_countries(date, model_path, training_data):
    """
    Predict revenue for all countries on a given date.

    Args:
        date (datetime): Prediction date.
        model_path (str): Path to model.
        training_data (pd.DataFrame): Training data for encoder and lag.

    Returns:
        tuple: (list of {'country': str, 'revenue': float}, float total_revenue)
    """
    model = load_model(model_path)
    countries = training_data['country'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(training_data['country'])

    year = date.year
    month = date.month
    times_viewed = 0.0
    input_data = []
    for country in countries:
        country_encoded = label_encoder.transform([country])[0]
        country_data = training_data[training_data['country'] == country].sort_values(
            'date')
        revenue_lag1 = country_data['revenue'].iloc[-1] if not country_data.empty else 0.0
        input_data.append({
            'country_encoded': country_encoded,
            'month': month,
            'year': year,
            'times_viewed': times_viewed,
            'revenue_lag1': revenue_lag1
        })

    input_df = pd.DataFrame(input_data)
    revenues = model.predict(input_df)

    predictions = [
        {'country': country, 'revenue': float(revenue)}
        for country, revenue in zip(countries, revenues)
    ]
    total_revenue = sum(revenue for revenue in revenues)

    return predictions, total_revenue


def train_model(data, model_path):
    required_cols = ['year', 'month', 'country',
                     'revenue', 'times_viewed', 'date']
    if not all(col in data.columns for col in required_cols):
        raise ValueError("Invalid CSV schema")
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    label_encoder = LabelEncoder()
    data['country_encoded'] = label_encoder.fit_transform(data['country'])
    data['revenue_lag1'] = data.groupby(
        'country')['revenue'].shift(1).fillna(0)
    features = ['country_encoded', 'month',
                'year', 'times_viewed', 'revenue_lag1']
    X = data[features]
    y = data['revenue']
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42)
    if len(data) < 2:
        model.fit(X, y)
        rmse = 0.0
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        # Save RMSE to metrics file
        metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({'rmse': rmse}, f)
    except OSError as e:
        raise OSError(f"Failed to save model: {str(e)}")
    return rmse
