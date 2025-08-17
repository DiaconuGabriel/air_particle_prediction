from selenium_scrape import *
from model_trainer import *

# scrape_aqicn_data()
split_results, final_results, predictions_last7 = train_and_predict_temporal("D:/repo_github/cercetare/data/data.csv")