from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List, Dict, Any
import statistics
from collections import defaultdict
from datetime import date


app = FastAPI(
    title="Analytics API",
    description="API for calculating analytics based on datasets",
    version="1.0.0",
)


class FilteredEventData(BaseModel):
    time_object: Dict[str, Any]
    event_type: str
    attribute: Dict[str, Any]


class PredictionRequest(BaseModel):
    data: List[FilteredEventData]  # List of filtered event data
    x_attribute: str  # Name of the feature (x) attribute
    y_attribute: str  # Name of the target (y) attribute
    x_values: List[Any]  # List of x values to predict


class RequestBody(BaseModel):
    attribute_name: str
    data: List[FilteredEventData]


class FuturePrices(BaseModel):
    years: List[int]
    data: List[FilteredEventData]


@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # Extract x and y values from filtered event data based on provided attribute names
        x_data = [event.attribute.get(data.x_attribute) for event in data.data]
        y_data = [event.attribute.get(data.y_attribute) for event in data.data]

        # Convert data to numpy arrays
        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data)

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(x_data, y_data)

        # Prepare x_values for prediction (reshaping to match the model's expected input)
        x_values = np.array(x_data).reshape(-1, 1)

        # Make predictions
        prediction = model.predict(x_values)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/average-price-by-suburb")
def average_price_by_suburb(
    data: List[FilteredEventData],
):  # Modify the input to accept a list of FilteredEventData
    try:
        suburb_prices = {}

        for event in data:
            suburb = event.attribute.get("suburb")
            price = event.attribute.get("price")
            if suburb and price is not None:
                if suburb not in suburb_prices:
                    suburb_prices[suburb] = []
                suburb_prices[suburb].append(price)

        average_prices = {
            suburb: statistics.mean(prices) for suburb, prices in suburb_prices.items()
        }
        return {"average_prices": average_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/median-price-by-suburb")
def median_price_by_suburb(
    data: List[FilteredEventData],
):  # Modify the input to accept a list of FilteredEventData
    try:
        suburb_prices = {}

        for event in data:
            suburb = event.attribute.get("suburb")
            price = event.attribute.get("price")
            if suburb and price is not None:
                if suburb not in suburb_prices:
                    suburb_prices[suburb] = []
                suburb_prices[suburb].append(price)

        median_prices = {
            suburb: statistics.median(prices)
            for suburb, prices in suburb_prices.items()
        }
        return {"median_prices": median_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/highest-value")
def highest_value(data: RequestBody):
    try:
        # Extract the values of the specified attribute
        attribute_values = [
            event.attribute.get(data.attribute_name) for event in data.data
        ]

        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(
                status_code=400,
                detail="No valid values found for the specified attribute.",
            )

        # Find the highest value
        highest = max(attribute_values)

        return {"highest_value": highest}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/lowest-value")
def lowest_value(data: RequestBody):
    try:
        # Extract the values of the specified attribute
        attribute_values = [
            event.attribute.get(data.attribute_name) for event in data.data
        ]

        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(
                status_code=400,
                detail="No valid values found for the specified attribute.",
            )

        # Find the lowest value
        lowest = min(attribute_values)

        return {"lowest_value": lowest}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/median-value")
def median_value(data: RequestBody):
    try:
        # Extract the values of the specified attribute
        attribute_values = [
            event.attribute.get(data.attribute_name) for event in data.data
        ]

        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(
                status_code=400,
                detail="No valid values found for the specified attribute.",
            )

        # Find the median value
        median = statistics.median(attribute_values)

        return {"median_value": median}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-future-prices")
def predict_future_prices(data: FuturePrices):
    try:
        # Extract x (year) and y (price)
        x_data = []
        y_data = []

        for event in data.data:
            timestamp = event.time_object.get("timestamp")
            price = event.attribute.get("price")
            if timestamp and price:
                year = int(timestamp[:4])  # Extract date part
                x_data.append(year)
                y_data.append(price)

        if len(x_data) < 2:
            raise HTTPException(
                status_code=400, detail="Not enough data for prediction."
            )

        # Convert data to numpy arrays
        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x_data, y_data)

        # Predict for future years
        future_years = np.array(data.years).reshape(-1, 1)
        predictions = model.predict(future_years)

        return {
            "predicted_prices": dict(
                zip([int(year) for year in data.years], predictions.tolist())
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/price-outliers")
def price_outliers(data: List[FilteredEventData]):
    try:
        prices = [
            event.attribute.get("price")
            for event in data
            if event.attribute.get("price") is not None
        ]

        if len(prices) < 4:
            raise HTTPException(
                status_code=400, detail="Not enough data to calculate outliers."
            )

        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [
            price for price in prices if price < lower_bound or price > upper_bound
        ]

        return {"outliers": outliers}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/total-sales-per-year")
def total_sales_per_year(data: List[FilteredEventData]):
    try:
        sales_by_year = defaultdict(int)

        for event in data:
            timestamp = event.time_object.get("timestamp")
            if timestamp:
                year = date.fromisoformat(timestamp).year
                sales_by_year[year] += 1

        return {"total_sales_per_year": sales_by_year}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/most-expensive-and-cheapest-suburb")
def most_expensive_and_cheapest_suburb(data: List[FilteredEventData]):
    try:
        suburb_prices = {}

        for event in data:
            suburb = event.attribute.get("suburb")
            price = event.attribute.get("price")
            if suburb and price:
                if suburb in suburb_prices:
                    suburb_prices[suburb].append(price)
                else:
                    suburb_prices[suburb] = [price]

        avg_prices = {
            suburb: sum(prices) / len(prices)
            for suburb, prices in suburb_prices.items()
        }

        most_expensive = max(avg_prices, key=avg_prices.get)
        cheapest = min(avg_prices, key=avg_prices.get)

        return {"most_expensive_suburb": most_expensive, "cheapest_suburb": cheapest}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
