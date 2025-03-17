from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List, Dict, Any
import statistics


app = FastAPI(title="Analytics API", description="API for calculating analytics based on datasets", version="1.0.0")

class FilteredEventData(BaseModel):
    time_object: Dict[str, Any]
    event_type: str
    attribute: Dict[str, Any]

class PredictionRequest(BaseModel):
    x: list[float]
    y: list[float]
    predict: list[float]

class HouseSalesData(BaseModel):
    prices: List[float]
    suburbs: List[str]

@app.post('/predict')
def predict(
    data: List[FilteredEventData],  # List of filtered event data
    x_attribute: str,               # Name of the feature (x) attribute
    y_attribute: str,               # Name of the target (y) attribute
    x_values: List[float]           # List of x values to predict
):
    try:
        # Extract x and y values from filtered event data based on provided attribute names
        x_data = [event.attribute.get(x_attribute, 0) for event in data]
        y_data = [event.attribute.get(y_attribute, 0) for event in data]
        
        # Convert data to numpy arrays
        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data)
        
        # Initialize and fit the model
        model = LinearRegression()
        model.fit(x_data, y_data)
        
        # Prepare x_values for prediction (reshaping to match the model's expected input)
        x_values = np.array(x_values).reshape(-1, 1)
        
        # Make predictions
        prediction = model.predict(x_values)
        
        return {'prediction': prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/predict-categorical')
def predict(x_attribute: str, y_attribute: str, x_values: List[Any], data: List[FilteredEventData]):
    try:
        # Initialize a label encoder to convert categorical variables into numeric values
        label_encoder = LabelEncoder()
        
        # Extract x and y values based on attribute names from the request
        x_data = [event.attribute.get(x_attribute, 0) for event in data]  # Extract feature (e.g., suburb)
        y_data = [event.attribute.get(y_attribute, 0) for event in data]  # Extract target (e.g., price)

        # Encode categorical features (e.g., suburb names) to numeric values
        x_data_encoded = label_encoder.fit_transform(x_data)

        # Convert data to numpy arrays
        x_data_encoded = np.array(x_data_encoded).reshape(-1, 1)  # Reshape for model input
        y_data = np.array(y_data)  # Target values (price)

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(x_data_encoded, y_data)

        # Prepare x_values for prediction (reshaping to match the model's expected input)
        x_values_encoded = label_encoder.transform(x_values)  # Encode input feature values
        x_values_encoded = np.array(x_values_encoded).reshape(-1, 1)

        # Make predictions
        prediction = model.predict(x_values_encoded)
        
        # Return prediction results
        return {'prediction': prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/average-price-by-suburb')
def average_price_by_suburb(data: List[FilteredEventData]):  # Modify the input to accept a list of FilteredEventData
    try:
        suburb_prices = {}
        
        for event in data:
            suburb = event.attribute.get('suburb')
            price = event.attribute.get('price')
            if suburb and price is not None:
                if suburb not in suburb_prices:
                    suburb_prices[suburb] = []
                suburb_prices[suburb].append(price)
        
        average_prices = {suburb: statistics.mean(prices) for suburb, prices in suburb_prices.items()}
        return {'average_prices': average_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/median-price-by-suburb')
def median_price_by_suburb(data: List[FilteredEventData]):  # Modify the input to accept a list of FilteredEventData
    try:
        suburb_prices = {}
        
        for event in data:
            suburb = event.attribute.get('suburb')
            price = event.attribute.get('price')
            if suburb and price is not None:
                if suburb not in suburb_prices:
                    suburb_prices[suburb] = []
                suburb_prices[suburb].append(price)
        
        median_prices = {suburb: statistics.median(prices) for suburb, prices in suburb_prices.items()}
        return {'median_prices': median_prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/highest-value')
def highest_value(attribute_name: str, data: List[FilteredEventData]):
    try:
        # Extract the values of the specified attribute
        attribute_values = [event.attribute.get(attribute_name) for event in data]
        
        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(status_code=400, detail="No valid values found for the specified attribute.")
        
        # Find the highest value
        highest = max(attribute_values)
        
        return {'highest_value': highest}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/lowest-value')
def lowest_value(attribute_name: str, data: List[FilteredEventData]):
    try:
        # Extract the values of the specified attribute
        attribute_values = [event.attribute.get(attribute_name) for event in data]
        
        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(status_code=400, detail="No valid values found for the specified attribute.")
        
        # Find the lowest value
        lowest = min(attribute_values)
        
        return {'lowest_value': lowest}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/median-value')
def median_value(attribute_name: str, data: List[FilteredEventData]):
    try:
        # Extract the values of the specified attribute
        attribute_values = [event.attribute.get(attribute_name) for event in data]
        
        # Ensure there are no None values and the list is not empty
        attribute_values = [value for value in attribute_values if value is not None]

        if not attribute_values:
            raise HTTPException(status_code=400, detail="No valid values found for the specified attribute.")
        
        # Find the median value
        median = statistics.median(attribute_values)
        
        return {'median_value': median}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))