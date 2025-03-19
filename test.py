import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Citirea datelor din CSV
data = pd.read_excel('D:/repo_github/cercetare/datetoate.xlsx')

# Selectarea caracteristicilor și a țintei
features = ['co2_min', 'co2_max', 'co2_median', 'co2_q1', 'co2_q3', 'co2_stdev', 'co2_count',
            'rh_min', 'rh_max', 'rh_median', 'rh_q1', 'rh_q3', 'rh_stdev', 'rh_count',
            'presiune_min', 'presiune_max', 'presiune_median', 'presiune_q1', 'presiune_q3', 'presiune_stdev', 'presiune_count',
            'temperatura_min', 'temperatura_max', 'temperatura_median', 'temperatura_q1', 'temperatura_q3', 'temperatura_stdev', 'temperatura_count',
            'o3_min', 'o3_max', 'o3_median', 'o3_q1', 'o3_q3', 'o3_stdev', 'o3_count',
            'pm1_min', 'pm1_max', 'pm1_median', 'pm1_q1', 'pm1_q3', 'pm1_stdev', 'pm1_count',
            'pm10_min', 'pm10_max', 'pm10_median', 'pm10_q1', 'pm10_q3', 'pm10_stdev', 'pm10_count']

median_features = [feature for feature in features if 'median' in feature]

x = data[features]
y = data['pm2.5_median']

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.16, random_state=42, shuffle=False)

# Antrenarea modelelor de regresie
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'XGBoost': XGBRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5,max_iter=10000),
}

test_sizes = np.arange(0.1, 0.7, 0.005)

# Stocarea rezultatelor
best_results = {name: {'test_size': None, 'MSE': float('inf'), 'R2': float('-inf')} for name in models.keys()}

# Iterarea prin diferite valori ale test_size
# for test_size1 in test_sizes:
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size1, random_state=42, shuffle=False)
    
#     for name, model in models.items():
#         # Antrenarea modelului
#         model.fit(X_train, y_train)
        
#         # Prezicerea pe setul de testare
#         y_pred = model.predict(X_test)
        
#         # Calcularea MSE și R2 pe setul de testare
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         # Actualizarea celor mai bune rezultate
#         if mse < best_results[name]['MSE']:
#             best_results[name]['test_size'] = test_size1
#             best_results[name]['MSE'] = mse
#             best_results[name]['R2'] = r2

# # Afișarea celor mai bune rezultate
# for name, result in best_results.items():
#     print(f"Model: {name}")
#     print(f"  Best test_size: {result['test_size']}")
#     print(f"  MSE: {result['MSE']}")
#     print(f"  R2: {result['R2']}")
#     print()

# Evaluarea și compararea modelelor
results = {}
for name, model in models.items():
    # Antrenarea modelului
    model.fit(X_train, y_train)
    
    # Prezicerea pe setul de testare
    y_pred = model.predict(X_test)
    
    # Calcularea MSE și R2 pe setul de testare
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse}, R2: {r2}")
    
    # Validarea încrucișată
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    mean_mse = mse_scores.mean()
    std_mse = mse_scores.std()
    # print(f"{name} - Cross-Validation Mean MSE: {mean_mse}, Std MSE: {std_mse}")
    
data_to_predict = pd.read_excel('D:/repo_github/cercetare/date_de_prezis.xlsx')

# Selectarea caracteristicilor pentru prezicere
x_to_predict = data_to_predict[features]

# Prezicerea pm2.5 cu fiecare model antrenat pentru noul set de date
predictions = {}
for name, model in models.items():
    y_pred_new = model.predict(x_to_predict)
    y_pred_new = np.round(y_pred_new, 0)
    predictions[name] = y_pred_new
    print(f"Predicțiile pentru modelul {name} pe noul set de date: {y_pred_new}")