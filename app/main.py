from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List
import statistics

app = FastAPI(title="Analytics API", description="API for calculating analytics based on datasets", version="1.0.0")

class PredictionRequest(BaseModel):
    x: list[float]
    y: list[float]
    predict: list[float]

class HouseSalesData(BaseModel):
    prices: List[float]
    suburbs: List[str]

@app.post('/predict')
def predict(data: PredictionRequest):
    try:
        x_values = np.array(data.x).reshape(-1, 1)
        y_values = np.array(data.y)
        
        model = LinearRegression()
        model.fit(x_values, y_values)
        
        prediction = model.predict(np.array(data.predict).reshape(-1, 1))
        return {'prediction': prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/average-price-by-suburb')
def average_price_by_suburb(data: HouseSalesData):
    try:
        suburb_prices = {}
        for suburb, price in zip(data.suburbs, data.prices):
            if suburb not in suburb_prices:
                suburb_prices[suburb] = []
            suburb_prices[suburb].append(price)
        
        average_prices = {suburb: statistics.mean(prices) for suburb, prices in suburb_prices.items()}
        return {'average_prices': average_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/median-price-by-suburb')
def median_price_by_suburb(data: HouseSalesData):
    try:
        suburb_prices = {}
        for suburb, price in zip(data.suburbs, data.prices):
            if suburb not in suburb_prices:
                suburb_prices[suburb] = []
            suburb_prices[suburb].append(price)
        
        median_prices = {suburb: statistics.median(prices) for suburb, prices in suburb_prices.items()}
        return {'median_prices': median_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/highest")
async def highest(attribute: str, event_type: str):
    # Dummy calculation (replace)
    highest_value = 999  
    return {"highest_value": highest_value}

@app.get("/lowest")
async def lowest(attribute: str, event_type: str):
    # Dummy calculation (replace)
    lowest_value = 1  
    return {"lowest_value": lowest_value}

