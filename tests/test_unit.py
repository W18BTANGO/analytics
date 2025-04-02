import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_missing_attribute():
    response = client.post("/predict", json={
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}],
        "x_attribute": "missing_x",
        "y_attribute": "missing_y",
        "x_values": [1, 2, 3]
    })
    assert response.status_code == 400
    assert "detail" in response.json()

def test_average_price_by_suburb_no_valid_data():
    response = client.post("/average-price-by-suburb", json=[
        {"time_object": {}, "event_type": "sale", "attribute": {}}
    ])
    assert response.status_code == 200
    assert response.json() == {"average_prices": {}}

def test_median_price_by_suburb_no_valid_data():
    response = client.post("/median-price-by-suburb", json=[
        {"time_object": {}, "event_type": "sale", "attribute": {}}
    ])
    assert response.status_code == 200
    assert response.json() == {"median_prices": {}}

def test_highest_value_no_valid_values():
    response = client.post("/highest-value", json={
        "attribute_name": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid values found" in response.json()["detail"]

def test_lowest_value_no_valid_values():
    response = client.post("/lowest-value", json={
        "attribute_name": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid values found" in response.json()["detail"]

def test_median_value_no_valid_values():
    response = client.post("/median-value", json={
        "attribute_name": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid values found" in response.json()["detail"]

def test_predict_future_prices_not_enough_data():
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": [{"time_object": {"timestamp": "2024-01-01"}, "event_type": "sale", "attribute": {"price": 500000}}]
    })
    assert response.status_code == 400
    assert "Not enough data for prediction" in response.json()["detail"]

def test_price_outliers_not_enough_data():
    response = client.post("/price-outliers", json=[
        {"time_object": {}, "event_type": "sale", "attribute": {"price": 100000}}
    ])
    assert response.status_code == 400
    assert "Not enough data to calculate outliers" in response.json()["detail"]

def test_total_sales_per_year_invalid_date():
    response = client.post("/total-sales-per-year", json=[
        {"time_object": {"timestamp": "invalid-date"}, "event_type": "sale", "attribute": {}}
    ])
    assert response.status_code == 400

def test_most_expensive_and_cheapest_suburb_no_data():
    response = client.post("/most-expensive-and-cheapest-suburb", json=[])
    assert response.status_code == 400
