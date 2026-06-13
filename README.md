 # Kvartis-bot 
 
### [English](https://github.com/zect-project/Kvartis-bot/blob/main/README_ENGLISH.md) | [Русский]([https://github.com/zect-project/Kvartis-bot/blob/main/README.md])

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**ИИ-бот для точной оценки рыночной стоимости квартир**  
Модель на **CatBoostRegressor**, обученная на реальных данных о недвижимости.  
Быстро и точно предсказывает цену по ключевым параметрам: город, количество комнат, площадь, ремонт, этаж и этажность дома.

---

##  Возможности

- Точная предсказательная модель (CatBoost + логарифмирование цены)
- Удобный веб-интерфейс (FastAPI + Jinja2)
- Автоматическая нормализация данных
- Сохранение и загрузка обученной модели
- Простая форма ввода параметров квартиры

---

##  Технологии

- **Лучшая версия python: 3.12**
- **CatBoost** - основная модель
- **FastAPI** + Jinja2 - веб-приложение
- **Pandas, NumPy, scikit-learn** - обработка данных
- **Joblib** - сохранение скейлера

---


##  Установка и запуск


### Клонирование
```bash
git clone https://github.com/zect-project/Kvartis-bot.git
cd Kvartis-bot
```
### Зависимости
pip install fastapi uvicorn catboost pandas numpy joblib scikit-learn jinja2

## Запуск веб-приложения
```
uvicorn main:app --reload
```
## Запуск Docker
```
docker compose down
docker compose up --build
```
#### Открой http://localhost:8000

## Веб интерфейс
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/interface_1.png)
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/interface_2.png)
