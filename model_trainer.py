import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_and_predict_temporal(data_path, target='PM2.5_median', test_sizes=np.arange(0.1, 0.7, 0.05)):
    # -----------------------------
    # Data read
    # -----------------------------
    data = pd.read_csv(data_path)
    
    # Features
    features = [col for col in data.columns if any(x in col for x in ['co2_median', 'R.H._median', 'Press_median',
                                                                    'Temp_median', 'O3_median', 'PM1_median', 'PM10_median'])]
    
    last_days_number = 50 # Unseed last days (for prediction)

    last_days = data.tail(last_days_number)
    older_data = data.iloc[:-last_days_number]

    X_older = older_data[features]
    y_older = older_data[target]

    X_last_days = last_days[features]
    y_last_days = last_days[target]

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'XGBoost': XGBRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    }

    # -----------------------------
    # Search for the best split on older data
    # -----------------------------
    split_results = {}

    for name, model in models.items():
        split_results[name] = {'test_size': None, 'MSE': np.inf, 'MAE': None, 'R2': None}
        for test_size in test_sizes:
            X_train, X_val, y_train, y_val = train_test_split(
                X_older, y_older, test_size=test_size, shuffle=False
            )

            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred_val)

            if mse < split_results[name]['MSE']:
                split_results[name] = {
                    'test_size': test_size,
                    'MSE': mse,
                    'MAE': mean_absolute_error(y_val, y_pred_val),
                    'R2': r2_score(y_val, y_pred_val)
                }

    # -----------------------------
    # Final training for best split for each model  
    # -----------------------------
    final_results = {}
    predictions_last = {}

    for name, model in models.items():
        best_test_size = split_results[name]['test_size']
        X_train, _, y_train, _ = train_test_split(
            X_older, y_older, test_size=best_test_size, shuffle=False
        )
        model.fit(X_train, y_train)
        y_pred_last_days = model.predict(X_last_days)
        predictions_last[name] = y_pred_last_days
        mae = mean_absolute_error(y_last_days, y_pred_last_days)
        mse = mean_squared_error(y_last_days, y_pred_last_days)
        r2 = r2_score(y_last_days, y_pred_last_days)
        accuracy_percent = np.mean((1 - np.abs((y_pred_last_days - y_last_days))/y_last_days) * 100)
        final_results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2, 'Accuracy %': accuracy_percent}

    # -----------------------------
    # Prediction plots for x days
    # -----------------------------
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8), sharex=True)

    model_names = list(models.keys())
    for idx, name in enumerate(model_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.plot(range(last_days_number), y_last_days, 'o-', label='Real', color='black')
        ax.plot(range(last_days_number), predictions_last[name], 'x--', label=f'{name} Predicted')

        # Calculating procentual error for each day
        eroare_procentuala = np.abs(predictions_last[name] - y_last_days) / np.abs(y_last_days) * 100
        worst_idx = np.argmax(eroare_procentuala)

        # Mark the worst prediction
        ax.plot(worst_idx, predictions_last[name][worst_idx], 'ro', markersize=10, label='Worst')

        ax.set_title(f'Split: {split_results[name]["test_size"]:.2f} {name} - Mean Accuracy: {final_results[name]["Accuracy %"]:.2f}%')
        ax.set_xlabel(f'Last {last_days_number} days')
        ax.set_ylabel(target)
        ax.legend()

    plt.tight_layout()
    plt.show()

    return split_results, final_results, predictions_last