 # Kvartis-bot 
 
### [English](https://github.com/zect-project/Kvartis-bot/blob/main/README_ENGLISH.md) | [Русский]([https://github.com/zect-project/Kvartis-bot/blob/main/README.md])

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


### 1. Клонирование
```bash
git clone https://github.com/zect-project/Kvartis-bot.git
cd Kvartis-bot
```
### 2. Зависимости
pip install fastapi uvicorn catboost pandas numpy joblib scikit-learn jinja2

### 3. Запуск веб-приложения
uvicorn main:app --reload

Открой в браузере: http://127.0.0.1:8000

## Веб интерфейс
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/web_interface_1.png)
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/web_interface_2.png)
