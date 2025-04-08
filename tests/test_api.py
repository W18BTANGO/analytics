from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test the API endpoints with valid data


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


def test_average_by_attribute():
    response = client.post("/average-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["average_values"]["Downtown"] == 550000


def test_median_by_attribute():
    response = client.post("/median-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 700000}},
            {"time_object": {"timestamp": "2023-08-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["median_values"]["Downtown"] == 600000


def test_highest_value():
    response = client.post("/highest-value", json={
        "attribute_name": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"price": 500000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"price": 700000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["highest_value"] == 700000


def test_lowest_value():
    response = client.post("/lowest-value", json={
        "attribute_name": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"price": 500000}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"price": 700000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["lowest_value"] == 500000


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "microservice": "analytics", "updated": "02/04/2025"}
