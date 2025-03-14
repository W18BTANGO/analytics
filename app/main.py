from fastapi import FastAPI, HTTPException, Query
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
async def highest(prices: List[float] = Query(None)):
    try:
        if not prices:
            raise HTTPException(status_code=400, detail="No prices provided.")
        
        highest_value = max(prices)
        return {"highest_value": highest_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/lowest")
async def lowest(prices: List[float] = Query(None)):
    try:
        if not prices:
            raise HTTPException(status_code=400, detail="No prices provided.")
        
        lowest_value = min(prices)
        return {"lowest_value": lowest_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/median")
async def median(prices: List[float] = Query(None)):
    try:
        if not prices:
            raise HTTPException(status_code=400, detail="No prices provided.")
        
        median_value = statistics.median(prices)
        return {"median": median_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



