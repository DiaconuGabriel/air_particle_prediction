This project focuses on predicting PM2.5 levels, using multiple machine learning models. The goal is to leverage environmental data such as CO2, temperature, humidity, and pressure to predict PM2.5 levels and to see which ml model does the best work. I explored different data splits for training and testing to find an optimal value that yields consistent and accurate predictions across all models.

Key Features:

Data Collection: Environmental data (CO2, temperature, humidity, etc.) is read from an Excel file. Preprocessing steps include selecting relevant features, and preparing the data for model training.
Machine Learning Models: Implemented models include Random Forest, XGBoost, Gradient Boosting, Decision Trees, and ElasticNet to predict PM2.5 levels.
Model Evaluation: Models are evaluated using metrics such as MSE and R2, performed for more robust performance estimation.
Prediction: PM2.5 levels are predicted for new environmental data.

Technologies Used:

Python
scikit-learn (for model training and evaluation)
XGBoost (for gradient boosting)
pandas (for data handling and preprocessing)

![image](https://github.com/user-attachments/assets/e9f53b09-f8e5-48f9-831d-3d92c0eb6bcb)
