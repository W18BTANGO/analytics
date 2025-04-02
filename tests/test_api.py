import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ...existing code for API endpoint tests...

def test_predict():
    response = client.post("/predict", json={
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"sqft": 1500, "price": 300000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"sqft": 2000, "price": 400000}}
        ],
        "x_attribute": "sqft",
        "y_attribute": "price",
        "x_values": [1800]
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_average_price_by_suburb():
    response = client.post("/average-price-by-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ])
    assert response.status_code == 200
    assert response.json()["average_prices"]["Downtown"] == 550000

def test_median_price_by_suburb():
    response = client.post("/median-price-by-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 700000}},
        {"time_object": {"timestamp": "2023-08-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ])
    assert response.status_code == 200
    assert response.json()["median_prices"]["Downtown"] == 600000

# ...other API tests for endpoints like /highest-value, /lowest-value, etc...

def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == { "status": "healthy","microservice":"analytics" }
