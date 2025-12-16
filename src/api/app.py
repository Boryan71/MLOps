from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Подгружаем модель
MODEL_PATH = "models/LinearRegr.pkl"
model = joblib.load(MODEL_PATH)

# Определяем схему входных данныx
class InputData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

app = FastAPI()

@app.post("/predict/")
def predict(input_data: InputData):
    try:
        # Формируем датафрейм из полученных данных
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Выдаем предсказание
        pred = model.predict_proba(df)[0][1]
        return f"Вероятность дефолта: {pred*100:.2f}%"
    except Exception as e:
        return {"error": str(e)}