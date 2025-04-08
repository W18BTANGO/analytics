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

def test_average_by_attribute_no_valid_data():
    response = client.post("/average-by-attribute", json={
        "group_by_attribute": "category",
        "value_attribute": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid data found" in response.json()["detail"]

def test_median_by_attribute_no_valid_data():
    response = client.post("/median-by-attribute", json={
        "group_by_attribute": "category",
        "value_attribute": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid data found" in response.json()["detail"]

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

def test_predict_future_values_not_enough_data():
    response = client.post("/predict-future-values", json={
        "time_points": [2025],
        "value_attribute": "price",
        "data": [{"time_object": {"timestamp": "2024-01-01"}, "event_type": "sale", "attribute": {"price": 500000}}]
    })
    assert response.status_code == 400
    assert "Not enough data for prediction" in response.json()["detail"]

def test_outliers_not_enough_data():
    response = client.post("/outliers", json={
        "value_attribute": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {"price": 100000}}]
    })
    assert response.status_code == 400
    assert "Not enough data to calculate outliers" in response.json()["detail"]

def test_count_by_time_invalid_date():
    response = client.post("/count-by-time", json={
        "time_format": "year",
        "data": [{"time_object": {"timestamp": "invalid-date"}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400

def test_min_max_by_attribute_no_data():
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "category",
        "value_attribute": "price",
        "data": []
    })
    assert response.status_code == 400
