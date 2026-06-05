# Kvartis-bot

### [English]([https://github.com/zect-project/Kvartis-bot/blob/main/README_ENGLISH.md]) | [Русский](https://github.com/zect-project/Kvartis-bot/blob/main/README.md)

**AI Bot for Accurate Apartment Market Value Estimation**
A **CatBoostRegressor** model trained on real estate data.  
Quickly and accurately predicts apartment prices based on key parameters: city, number of rooms, area, renovation quality, floor, and building floor count.

---

## Features
- High-precision prediction model (CatBoost + price log transformation)
- User-friendly web interface (FastAPI + Jinja2)
- Automatic data normalization
- Save and load trained model
- Simple input form for apartment parameters

---

## Technologies
- **Python 3.12** (recommended)
- **CatBoost** — main ML model
- **FastAPI** + **Jinja2** — web application
- **Pandas, NumPy, scikit-learn** — data processing
- **Joblib** — scaler and model persistence

---

## Installation and Launch

### 1. Clone the repository
```bash
git clone https://github.com/zect-project/Kvartis-bot.git
cd Kvartis-bot
```
### 2. Install dependencies
pip install fastapi uvicorn catboost pandas numpy joblib scikit-learn jinja2
### 3. Run the web application
uvicorn main:app --reload

Open in your browser: http://127.0.0.1:8000

## Web interface
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/web_interface_1.png)
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/web_interface_2.png)
