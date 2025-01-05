import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Citirea datelor din CSV
data = pd.read_excel('C:/Users/pc/Desktop/date/datetoate.xlsx')

# Selectarea caracteristicilor și a țintei
features = ['co2_min', 'co2_max', 'co2_median', 'co2_q1', 'co2_q3', 'co2_stdev', 'co2_count',
            'rh_min', 'rh_max', 'rh_median', 'rh_q1', 'rh_q3', 'rh_stdev', 'rh_count',
            'presiune_min', 'presiune_max', 'presiune_median', 'presiune_q1', 'presiune_q3', 'presiune_stdev', 'presiune_count',
            'temperatura_min', 'temperatura_max', 'temperatura_median', 'temperatura_q1', 'temperatura_q3', 'temperatura_stdev', 'temperatura_count',
            'o3_min', 'o3_max', 'o3_median', 'o3_q1', 'o3_q3', 'o3_stdev', 'o3_count',
            'pm1_min', 'pm1_max', 'pm1_median', 'pm1_q1', 'pm1_q3', 'pm1_stdev', 'pm1_count',
            'pm10_min', 'pm10_max', 'pm10_median', 'pm10_q1', 'pm10_q3', 'pm10_stdev', 'pm10_count']
x = data[features]
y = data['pm2.5_median']

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Antrenarea modelelor de regresie
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor()
}

# Evaluarea și compararea modelelor
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse}, R2: {r2}")

# Vizualizarea importanței caracteristicilor pentru RandomForest
rf_model = models['RandomForest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotarea importanței caracteristicilor
plt.figure()
plt.title("Importanța caracteristicilor - RandomForest")
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
plt.xlim([-1, x.shape[1]])
plt.show()
