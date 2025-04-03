from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression  # Ensure sklearn is installed
import numpy as np
from typing import List, Dict, Any, Optional
import statistics
from collections import defaultdict
from datetime import date
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Analytics API",
    description="API for calculating analytics based on datasets",
    version="1.0.0",
)


class FilteredEventData(BaseModel):
    time_object: Dict[str, Any]
    event_type: str
    attribute: Dict[str, Any]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class PredictionRequest(BaseModel):
    data: List[FilteredEventData]  # List of filtered event data
    x_attribute: str  # Name of the feature (x) attribute
    y_attribute: str  # Name of the target (y) attribute
    x_values: List[float]  # List of x values to predict


class RequestBody(BaseModel):
    attribute_name: str
    data: List[FilteredEventData]


class FuturePrices(BaseModel):
    years: List[int]
    data: List[FilteredEventData]


@app.post("/predict")
def predict(data: PredictionRequest) -> Dict[str, List[float]]:
    try:
        # Extract x and y values from filtered event data based on provided attribute names
        x_data_raw: List[Optional[Any]] = [event.attribute.get(data.x_attribute) for event in data.data]
        y_data_raw: List[Optional[Any]] = [event.attribute.get(data.y_attribute) for event in data.data]

        # Filter out None values
        x_data_filtered = [x for x in x_data_raw if x is not None]
        y_data_filtered = [y for y in y_data_raw if y is not None]

        if len(x_data_filtered) != len(y_data_filtered):
            raise HTTPException(status_code=400, detail="Mismatched x and y data lengths.")

        # Convert data to numpy arrays
        x_data: np.ndarray = np.array(x_data_filtered, dtype=float).reshape(-1, 1)
        y_data: np.ndarray = np.array(y_data_filtered, dtype=float)

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(x_data, y_data)

        # Prepare x_values for prediction (reshaping to match the model's expected input)
        x_values: np.ndarray = np.array(data.x_values, dtype=float).reshape(-1, 1)

        # Make predictions
        prediction = model.predict(x_values)

        return {"prediction": prediction.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/average-price-by-suburb")
def average_price_by_suburb(data: List[FilteredEventData]) -> Dict[str, Dict[str, float]]:
    try:
        suburb_prices: Dict[str, List[float]] = {}

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
def median_price_by_suburb(data: List[FilteredEventData]) -> Dict[str, Dict[str, float]]:
    try:
        suburb_prices: Dict[str, List[float]] = {}

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
def highest_value(data: RequestBody) -> Dict[str, float]:
    try:
        # Extract the values of the specified attribute
        attribute_values: List[float] = [
            value for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
        ]

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
def lowest_value(data: RequestBody) -> Dict[str, float]:
    try:
        # Extract the values of the specified attribute
        attribute_values: List[float] = [
            value for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
        ]

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
def median_value(data: RequestBody) -> Dict[str, float]:
    try:
        # Extract the values of the specified attribute
        attribute_values: List[float] = [
            value for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
        ]

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
def predict_future_prices(data: FuturePrices) -> Dict[str, Dict[int, float]]:
    try:
        # Extract x (year) and y (price)
        x_data: List[int] = []
        y_data: List[float] = []

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
        x_data_np = np.array(x_data, dtype=float).reshape(-1, 1)
        y_data_np = np.array(y_data, dtype=float)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x_data_np, y_data_np)

        # Predict for future years
        future_years_np = np.array(data.years, dtype=float).reshape(-1, 1)
        predictions = model.predict(future_years_np)

        return {
            "predicted_prices": dict(
                zip([int(year) for year in data.years], predictions.tolist())
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/price-outliers")
def price_outliers(data: List[FilteredEventData]) -> Dict[str, List[float]]:
    try:
        # Extract prices from the input data
        prices: List[float] = [
            price for price in (event.attribute.get("price") for event in data) if price is not None
        ]

        # Ensure there are enough data points to calculate outliers
        if len(prices) < 4:
            raise HTTPException(
                status_code=400, detail="Not enough data to calculate outliers."
            )

        # Calculate the interquartile range (IQR)
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1

        # Calculate bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = [
            price for price in prices if price < lower_bound or price > upper_bound
        ]

        return {"outliers": outliers}

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Error in calculating percentiles: " + str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/total-sales-per-year")
def total_sales_per_year(data: List[FilteredEventData]) -> Dict[str, Dict[int, int]]:
    try:
        sales_by_year: Dict[int, int] = defaultdict(int)

        for event in data:
            timestamp = event.time_object.get("timestamp")
            if timestamp:
                year = date.fromisoformat(timestamp).year
                sales_by_year[year] += 1

        return {"total_sales_per_year": sales_by_year}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/most-expensive-and-cheapest-suburb")
def most_expensive_and_cheapest_suburb(data: List[FilteredEventData]) -> Dict[str, str]:
    try:
        suburb_prices: Dict[str, List[float]] = {}

        for event in data:
            suburb = event.attribute.get("suburb")
            price = event.attribute.get("price")
            if suburb and price is not None:
                if suburb not in suburb_prices:
                    suburb_prices[suburb] = []
                suburb_prices[suburb].append(price)

        avg_prices = {
            suburb: sum(prices) / len(prices)
            for suburb, prices in suburb_prices.items()
        }

        # Use a lambda function to extract values for comparison
        most_expensive = max(avg_prices, key=lambda suburb: avg_prices[suburb])
        cheapest = min(avg_prices, key=lambda suburb: avg_prices[suburb])

        return {"most_expensive_suburb": most_expensive, "cheapest_suburb": cheapest}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "microservice": "analytics", "updated": "02/04/2025"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)  # Change port number here
