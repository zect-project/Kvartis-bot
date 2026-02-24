from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler 
from catboost import CatBoostRegressor 
from catboost import Pool  
import pandas as pd   
import numpy as np 
import joblib

model = CatBoostRegressor(   # сама модель
    iterations=4000,
    learning_rate=0.25,
    depth=8,
    loss_function='RMSE',   # для log-price это нормально
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds = 300
)

model.load_model('C:/holl/python/Kvartis/models/kvartis_model.cbm')
scaler = joblib.load('C:/holl/python/Kvartis/models/kvartis_scaler.pkl')
df = pd.read_csv('C:/holl/python/Kvartis/web/wdata.csv', header=None)  
df.columns = ['city', 'rooms', 'm2', 'repair', 'floor', 'all_floor']

row = df.iloc[0]
x = pd.DataFrame([row])

numeric = ['rooms', 'm2', 'floor', 'all_floor']
x[numeric] = scaler.transform(x[numeric])

pred_log = model.predict(x)
pred_price = np.expm1(pred_log)[0]

print(x.to_string(index=False))
print("цена ≈", f"{pred_price:,.0f}", "₽")