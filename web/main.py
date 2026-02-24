from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import csv
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CSV_FILE = "wdata.csv"

# Создаём файл с заголовками, если его ещё нет
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["city", "rooms", "m2", "repair", "floor", "all_floor"])

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": ""}
    )

@app.post("/save", response_class=HTMLResponse)
async def save_apartment(
    request: Request,
    city: str = Form(...),
    rooms: int = Form(...),
    m2: int = Form(...),          
    repair: str = Form(...),
    floor: int = Form(...),
    all_floor: int = Form(...),
):
    row = [city, rooms, m2, repair, floor, all_floor]
    
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": "Данные успешно сохранены!"}
    )