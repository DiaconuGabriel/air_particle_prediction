# Air Quality Research - Machine Learning & Selenium

This project focuses on predicting PM2.5 levels using multiple machine learning models. The goal is to leverage environmental data such as CO2, temperature, humidity, and pressure to predict PM2.5 levels and to see which ML model performs best. The project explores different data splits for training and testing to find an optimal value that yields consistent and accurate predictions across all models.

## Features

- Automated web scraping with Selenium to download historical pollutant data (CO2, R.H., Press, Temp, O3, PM1, PM10, PM2.5) for a specific location from [aqicn.org](https://aqicn.org).
- Data processing and saving in CSV format.
- Training and evaluation of several regression models (RandomForest, LinearRegression, DecisionTree, XGBoost, GradientBoosting, ElasticNet) for PM2.5 prediction.
- Exploration of different train/test splits to optimize model performance.
- Performance metrics: MSE, MAE, R2, percentage accuracy, mean absolute percentage error (MAPE).
- Comparative plots between real and predicted values for the last days, highlighting the worst prediction.

## Requirements

See `requirements.txt` for all dependencies.  
Install them with:

```bash
pip install -r requirements.txt
```

You also need [ChromeDriver](https://chromedriver.chromium.org/downloads) compatible with your Chrome version.

## How to Run

1. **Clone the repo and install dependencies:**
    ```bash
    git clone https://github.com/username/cercetare.git
    cd cercetare
    pip install -r requirements.txt
    ```

2. **Create a `.env` file** in the project root with your ChromeDriver path:
    ```
    chrome_driver_path=D:/path/to/chromedriver.exe
    ```

3. **Run the main script:**
    ```bash
    python automation.py
    ```

   - To collect new data, uncomment the `scrape_aqicn_data()` line in `automation.py`.

## Project Structure

- **selenium_scrape.py** – collects and saves data from the website.
- **model_trainer.py** – trains and evaluates regression models, displays plots and metrics.
- **automation.py** – automation script for running the full workflow

## Example Metrics

- **MAE** (Mean Absolute Error): average absolute error (units)
- **MSE** (Mean Squared Error): mean squared error
- **R2**: coefficient of determination (1 = perfect)
- **Accuracy %**: percentage closeness to the real value

## Charts for seeing how models perform

For the last 7 days at the time runned: 17/08/2025

<img width="1800" height="800" alt="7daysbest" src="https://github.com/user-attachments/assets/cb761d0d-a6ad-4aa4-90fe-96d5c31db0a7" />

For the last 50 days at the time runned: 17/08/2025

<img width="1800" height="800" alt="50daysbest" src="https://github.com/user-attachments/assets/b9cc86b3-f728-46ac-992d-0f404011cb0d" />

For the last 100 days at the time runned: 17/08/2025

<img width="1800" height="800" alt="100daysbest" src="https://github.com/user-attachments/assets/be619eff-3624-4000-a3b7-81ba5dd27dbd" />

