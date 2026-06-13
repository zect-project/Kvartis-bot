# Kvartis-bot

### [English]([https://github.com/zect-project/Kvartis-bot/blob/main/README_ENGLISH.md]) | [Русский](https://github.com/zect-project/Kvartis-bot/blob/main/README.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

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

### Clone the repository
```bash
git clone https://github.com/zect-project/Kvartis-bot.git
cd Kvartis-bot
```
### Install dependencies
```
pip install fastapi uvicorn catboost pandas numpy joblib scikit-learn jinja2
```
### Run with Uvicorn
```
uvicorn main:app --reload
```
### Run with Docker
```
docker compose down
docker compose up --build
```
#### Open in your browser: http://localhost:8000

## Web interface
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/interface_1.png)
![](https://github.com/zect-project/Kvartis-bot/blob/main/image/interface_2.png)
