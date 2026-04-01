Markdown# Kvartis-bot 🏙️💸

**ИИ-бот для точной оценки рыночной стоимости квартир**  
Модель на **CatBoostRegressor**, обученная на реальных данных о недвижимости.  
Быстро и точно предсказывает цену по ключевым параметрам: город, количество комнат, площадь, ремонт, этаж и этажность дома.

---

## ✨ Возможности

- Точная предсказательная модель (CatBoost + логарифмирование цены)
- Удобный веб-интерфейс (FastAPI + Jinja2)
- Автоматическая нормализация данных
- Сохранение и загрузка обученной модели
- Простая форма ввода параметров квартиры

---

## 🛠 Технологии

- **Лучше всего подходит Python 3.12**
- **CatBoost** - основная модель
- **FastAPI** + Jinja2 - веб-приложение
- **Pandas, NumPy, scikit-learn** - обработка данных
- **Joblib** - сохранение скейлера

---


## 🚀 Установка и запуск

```bash
### 1. Клонирование
git clone https://github.com/zect-project/Kvartis-bot.git
cd Kvartis-bot

2. Зависимости
pip install fastapi uvicorn catboost pandas numpy joblib scikit-learn jinja2

3. Исправление путей (обязательно!)
В файле web/main.py замени жёстко заданные пути на относительные:
Python# Было:
model.load_model('C:/holl/python/Kvartis/models/kvartis_model.cbm')
scaler = joblib.load('C:/holl/python/Kvartis/models/kvartis_scaler.pkl')
CSV_PATH = 'C:/holl/python/Kvartis/web/wdata.csv'

Запуск веб-приложения
uvicorn main:app --reload
Открой в браузере: http://127.0.0.1:8000
