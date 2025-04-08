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


class AggregateByAttributeRequest(BaseModel):
    group_by_attribute: str
    value_attribute: str
    data: List[FilteredEventData]


class FutureValuesRequest(BaseModel):
    time_points: List[int]
    value_attribute: str
    data: List[FilteredEventData]


class OutliersRequest(BaseModel):
    value_attribute: str
    data: List[FilteredEventData]


class MinMaxByAttributeRequest(BaseModel):
    group_by_attribute: str
    value_attribute: str
    data: List[FilteredEventData]


class CountByTimeRequest(BaseModel):
    time_format: str = "year"  # year, month, day
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

        if not x_data_filtered or not y_data_filtered:
            raise HTTPException(status_code=400, detail="Input data cannot be empty.")

        if len(x_data_filtered) != len(y_data_filtered):
            raise HTTPException(status_code=400, detail="Mismatched x and y data lengths.")

        # Convert data to numpy arrays
        x_data: np.ndarray = np.array(x_data_filtered, dtype=float).reshape(-1, 1)
        y_data: np.ndarray = np.array(y_data_filtered, dtype=float)

        if np.isnan(x_data).any() or np.isnan(y_data).any():
            raise HTTPException(status_code=400, detail="Input data contains NaN or invalid values.")

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


@app.post("/average-by-attribute")
def average_by_attribute(data: AggregateByAttributeRequest) -> Dict[str, Dict[str, float]]:
    try:
        grouped_values: Dict[str, List[float]] = {}

        for event in data.data:
            group_key = event.attribute.get(data.group_by_attribute)
            value = event.attribute.get(data.value_attribute)
            if group_key is not None and value is not None:
                if group_key not in grouped_values:
                    grouped_values[group_key] = []
                grouped_values[group_key].append(float(value))

        if not grouped_values:
            raise HTTPException(
                status_code=400,
                detail=f"No valid data found for attributes: {data.group_by_attribute}, {data.value_attribute}",
            )

        average_values = {
            group: statistics.mean(values) for group, values in grouped_values.items()
        }
        return {"average_values": average_values}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/median-by-attribute")
def median_by_attribute(data: AggregateByAttributeRequest) -> Dict[str, Dict[str, float]]:
    try:
        grouped_values: Dict[str, List[float]] = {}

        for event in data.data:
            group_key = event.attribute.get(data.group_by_attribute)
            value = event.attribute.get(data.value_attribute)
            if group_key is not None and value is not None:
                if group_key not in grouped_values:
                    grouped_values[group_key] = []
                grouped_values[group_key].append(float(value))

        if not grouped_values:
            raise HTTPException(
                status_code=400,
                detail=f"No valid data found for attributes: {data.group_by_attribute}, {data.value_attribute}",
            )

        median_values = {
            group: statistics.median(values)
            for group, values in grouped_values.items()
        }
        return {"median_values": median_values}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/highest-value")
def highest_value(data: RequestBody) -> Dict[str, float]:
    try:
        # Extract the values of the specified attribute
        attribute_values: List[float] = [
            float(value) for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
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
            float(value) for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
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
            float(value) for value in (event.attribute.get(data.attribute_name) for event in data.data) if value is not None
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


@app.post("/predict-future-values")
def predict_future_values(data: FutureValuesRequest) -> Dict[str, Dict[int, float]]:
    try:
        # If time_points is empty, return empty predictions
        if not data.time_points:
            return {"predicted_values": {}}
            
        # Extract x (time point) and y (value)
        x_data: List[int] = []
        y_data: List[float] = []

        for event in data.data:
            timestamp = event.time_object.get("timestamp")
            value = event.attribute.get(data.value_attribute)
            if timestamp and value is not None:
                time_point = int(timestamp[:4])  # Extract year part
                x_data.append(time_point)
                y_data.append(float(value))

        if len(x_data) < 2:
            raise HTTPException(
                status_code=400, detail="Not enough data for prediction: At least 2 data points required"
            )

        # Convert data to numpy arrays
        x_data_np = np.array(x_data, dtype=float).reshape(-1, 1)
        y_data_np = np.array(y_data, dtype=float)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x_data_np, y_data_np)

        # Predict for future time points
        future_points_np = np.array(data.time_points, dtype=float).reshape(-1, 1)
        predictions = model.predict(future_points_np)

        return {
            "predicted_values": dict(
                zip([int(point) for point in data.time_points], predictions.tolist())
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/outliers")
def outliers(data: OutliersRequest) -> Dict[str, List[float]]:
    try:
        # Extract values from the input data
        values: List[float] = [
            float(value) for value in (event.attribute.get(data.value_attribute) for event in data.data) if value is not None
        ]

        # Ensure there are enough data points to calculate outliers
        if len(values) < 4:
            raise HTTPException(
                status_code=400, detail="Not enough data to calculate outliers: At least 4 data points required"
            )

        # Calculate the interquartile range (IQR)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # Calculate bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outlier_values = [
            value for value in values if value < lower_bound or value > upper_bound
        ]

        return {"outliers": outlier_values}

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Error in calculating percentiles: " + str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/count-by-time")
def count_by_time(data: CountByTimeRequest) -> Dict[str, Dict[str, int]]:
    try:
        counts_by_time: Dict[str, int] = defaultdict(int)

        for event in data.data:
            timestamp = event.time_object.get("timestamp")
            if timestamp:
                if data.time_format == "year":
                    time_key = str(date.fromisoformat(timestamp.split("T")[0]).year)
                elif data.time_format == "month":
                    dt = date.fromisoformat(timestamp.split("T")[0])
                    time_key = f"{dt.year}-{dt.month:02d}"
                elif data.time_format == "day":
                    time_key = timestamp.split("T")[0]
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid time_format: Must be 'year', 'month', or 'day'"
                    )
                counts_by_time[time_key] += 1

        return {"counts_by_time": dict(counts_by_time)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/min-max-by-attribute")
def min_max_by_attribute(data: MinMaxByAttributeRequest) -> Dict[str, str]:
    try:
        grouped_values: Dict[str, List[float]] = {}

        for event in data.data:
            group_key = event.attribute.get(data.group_by_attribute)
            value = event.attribute.get(data.value_attribute)
            if group_key is not None and value is not None:
                if group_key not in grouped_values:
                    grouped_values[group_key] = []
                grouped_values[group_key].append(float(value))

        if not grouped_values:
            raise HTTPException(
                status_code=400,
                detail=f"No valid data found for attributes: {data.group_by_attribute}, {data.value_attribute}",
            )

        avg_values = {
            group: sum(values) / len(values)
            for group, values in grouped_values.items()
        }

        # Use a lambda function to extract values for comparison
        max_key = max(avg_values, key=lambda k: avg_values[k])
        min_key = min(avg_values, key=lambda k: avg_values[k])

        return {
            "maximum_attribute": max_key, 
            "minimum_attribute": min_key
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "microservice": "analytics", "updated": "02/04/2025"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)  # Change port number here
