import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_workflow_predict_and_outliers():
    # Step 1: Predict prices
    predict_response = client.post("/predict", json={
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"sqft": 1500, "price": 300000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"sqft": 2000, "price": 400000}}
        ],
        "x_attribute": "sqft",
        "y_attribute": "price",
        "x_values": [1800, 1900]  # Ensure x_values is valid
    })
    assert predict_response.status_code == 200
    predictions = predict_response.json().get("prediction", [])
    
    # Ensure predictions list is not empty and has the expected number of elements
    assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"

    # Step 2: Check for outliers in the predicted prices
    outliers_response = client.post("/price-outliers", json=[
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"price": predictions[0]}},
        {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"price": predictions[1]}},
        {"time_object": {"timestamp": "2023-08-01"}, "event_type": "sale", "attribute": {"price": 5000000}},  # Add more data points
        {"time_object": {"timestamp": "2023-09-01"}, "event_type": "sale", "attribute": {"price": 250000}}
    ])
    assert outliers_response.status_code == 200
    assert "outliers" in outliers_response.json()

def test_workflow_sales_and_median_price():
    # Step 1: Record sales data
    sales_data = [
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ]
    sales_response = client.post("/total-sales-per-year", json=sales_data)
    assert sales_response.status_code == 200
    assert sales_response.json()["total_sales_per_year"]["2023"] == 2

    # Step 2: Calculate median price for the suburb
    median_response = client.post("/median-price-by-suburb", json=sales_data)
    assert median_response.status_code == 200
    assert median_response.json()["median_prices"]["Downtown"] == 550000
