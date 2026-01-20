from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error   # метрики для конечного результата
from sklearn.model_selection import train_test_split   # разбиваем данные
from sklearn.preprocessing import StandardScaler   # нормализация масштаба признаков
from catboost import CatBoostRegressor   # сама модель 
from catboost import Pool   # оптимизация работы с категориальными признаками
import pandas as pd   # база
import numpy as np   # база


#################### ПЕРЕОБРАЗОВЫВАЕМ ДАННЫЕ ####################

df = pd.read_csv('E:/qpo/pytop/Kvartirs/data/main_data.csv')   # ПОМЕНЯЙТЕ НА СВОЁ !!! подключаемся к базе

x = df.drop(columns=['real_price'])
y = df.drop(columns=['city', 'rooms', 'm2', 'repair', 'floor', 'all_floor'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)   # разбиваем данные


features = ['rooms', 'm2', 'floor', 'all_floor']   # убераем city и repair

y_train_log = np.log1p(y_train.values.ravel())   # приводим 
y_test_log  = np.log1p(y_test.values.ravel())

scaler = StandardScaler()   # приводим данные в другой вид для модели
x_train[features] = scaler.fit_transform(x_train[features])     
x_test[features] = scaler.transform(x_test[features])       


cat_features = ['city', 'repair']   # оставляем только city и repair

train_pool = Pool(   # данная модель работает лучше с категориальными признаками
    data = x_train,   # если бы мы превели их в цифры то было бы хуже
    label = y_train_log,
    cat_features = cat_features, 
    feature_names = x_train.columns.tolist()
)

test_pool = Pool(
    data = x_test,
    label = y_test_log,
    cat_features = cat_features,       
    feature_names = x_test.columns.tolist()
)


###################### ГЛАВНАЯ ЧАСТЬ ######################


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model = CatBoostRegressor(   # сама модель
    iterations=4000,
    learning_rate=0.25,
    depth=8,
    loss_function='RMSE',           # для log-price это нормально
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds = 300
)



model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)

y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test_log)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE:   {mae:,.0f} ₽")
print(f"RMSE:  {rmse:,.0f} ₽")
print(f"MAPE:  {mape:.2f}%")
print(f"R²:    {r2:.4f}")
