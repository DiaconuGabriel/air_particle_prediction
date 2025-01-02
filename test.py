import pandas as pd
from pycaret.classification import *

def categorize_pm25(value):
    if value < 12.5:
        return 'Good'
    elif 12.5 <= value < 25:
        return 'Fair'
    elif 25 <= value < 50:
        return 'Poor'
    elif 50 <= value < 150:
        return 'Very poor'
    else:
        return 'Extremely poor'
    
pm25 = pd.read_excel('C:/Users/pc/Desktop/pm2.5.xlsx')
# print(pm25)

pm25['air_quality_category'] = pm25['median'].apply(categorize_pm25)

print(pm25['air_quality_category'].value_counts())

clf1 = setup(data=pm25, target='air_quality_category', use_gpu=True)

best = compare_models()

# evaluate_model(best)

plot_model(best, plot = 'confusion_matrix')
# plot_model(best, plot = 'auc')
plot_model(best, plot = 'feature')