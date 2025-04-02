import pytest
from fastapi.testclient import TestClient
from app.main import app  # Assuming your FastAPI app is in main.py

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "data": [
            {
                "time_object": {
                    "timestamp": "2023-06-01T00:00:00",
                    "duration": 1,
                    "duration_unit": "second",
                    "timezone": "GMT+11"
                },
                "event_type": "sale",
                "attribute": {"sqft": 1500, "price": 300000}
            },
            {
                "time_object": {
                    "timestamp": "2023-07-01T00:00:00",
                    "duration": 1,
                    "duration_unit": "second",
                    "timezone": "GMT+11"
                },
                "event_type": "sale",
                "attribute": {"sqft": 2000, "price": 400000}
            }
        ],
        "x_attribute": "sqft",
        "y_attribute": "price",
        "x_values": [1800]
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_average_price_by_suburb():
    response = client.post("/average-price-by-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ])
    assert response.status_code == 200
    assert response.json()["average_prices"]["Downtown"] == 550000

def test_median_price_by_suburb():
    response = client.post("/median-price-by-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 700000}},
        {"time_object": {"timestamp": "2023-08-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ])
    assert response.status_code == 200
    assert response.json()["median_prices"]["Downtown"] == 600000

def test_highest_value():
    response = client.post("/highest-value", json={
        "attribute_name": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 700000}},
        ]
    })
    print(response.json())
    assert response.status_code == 200
    assert response.json()["highest_value"] == 700000

def test_lowest_value():
    response = client.post("/lowest-value", json={
        "attribute_name": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 700000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["lowest_value"] == 500000

def test_total_sales_per_year():
    response = client.post("/total-sales-per-year", json=[
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {}},
        {"time_object": {"timestamp": "2023-08-15"}, "event_type": "sale", "attribute": {}},
        {"time_object": {"timestamp": "2022-05-20"}, "event_type": "sale", "attribute": {}}
    ])
    print(response.json())
    assert response.status_code == 200
    assert response.json()["total_sales_per_year"]["2023"] == 2
    assert response.json()["total_sales_per_year"]["2022"] == 1

def test_price_outliers():
    response = client.post("/price-outliers", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 100000}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 200000}},
        {"time_object": {"timestamp": "2023-08-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 5000000}},
        {"time_object": {"timestamp": "2023-09-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 250000}}
    ])
    assert response.status_code == 200
    assert 5000000 in response.json()["outliers"]

def test_most_expensive_and_cheapest_suburb():
    response = client.post("/most-expensive-and-cheapest-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 500000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Rhodes", "price": 300000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Rhodes", "price": 350000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Darlinghurst", "price": 900000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Darlinghurst", "price": 1000000}}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result['most_expensive_suburb'] == 'Darlinghurst'
    assert result['cheapest_suburb'] == 'Rhodes'

# Test case 2: Only one suburb with multiple prices
def test_single_suburb():
    response = client.post("/most-expensive-and-cheapest-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result['most_expensive_suburb'] == 'Balmain'
    assert result['cheapest_suburb'] == 'Balmain'

# Test case 3: Data with missing suburb or price
def test_missing_suburb_or_price():
    # Missing suburb
    response = client.post("/most-expensive-and-cheapest-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 500000}},
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result['most_expensive_suburb'] == 'Balmain'
    assert result['cheapest_suburb'] == 'Balmain'

    # Missing price
    response = client.post("/most-expensive-and-cheapest-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown"}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result['most_expensive_suburb'] == 'Downtown'
    assert result['cheapest_suburb'] == 'Downtown'

# Test case 4: Multiple suburbs with equal average prices
def test_equal_average_prices():
    response = client.post("/most-expensive-and-cheapest-suburb", json=[
        {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Uptown", "price": 500000}}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result['most_expensive_suburb'] == 'Downtown'  # Or 'Rhodes' depending on the implementation logic
    assert result['cheapest_suburb'] == 'Downtown'  # Or 'Rhodes' depending on the implementation logic

# Test case 5: Edge case - No data
def test_no_data():
    response = client.post("/most-expensive-and-cheapest-suburb", json=[])
    assert response.status_code == 400  # Expecting a 400 error for no data
    assert "detail" in response.json()  # Should contain error message

# Test case 6: Handling error (e.g., invalid input format)
def test_invalid_input_format():
    response = client.post("/most-expensive-and-cheapest-suburb", json="invalid_data")
    assert response.status_code == 422
    assert "detail" in response.json()  # Should contain error message

def test_predict_future_prices():
    response = client.post("/predict-future-prices", json={
        "years": [2025, 2026, 2027],
        "data": [
            {"time_object": {"timestamp": "2020-06-01"}, "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01"},  "event_type": "sale", "attribute": {"price": 350000}},
            {"time_object": {"timestamp": "2022-06-01"},  "event_type": "sale", "attribute": {"price": 400000}},
            {"time_object": {"timestamp": "2023-06-01"},  "event_type": "sale", "attribute": {"price": 450000}},
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert "predicted_prices" in result
    predicted_prices = result["predicted_prices"]
    assert len(predicted_prices) == 3  # Should return predictions for 3 future years
    assert all(year in predicted_prices for year in ['2025', '2026', '2027'])

# Test case 2: Insufficient data (less than 2 data points)
def test_insufficient_data():
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": [
            {"time_object": {"timestamp": "2020-06-01T00:00:00"},  "event_type": "sale", "attribute": {"price": 300000}},
        ]
    })
    assert response.status_code == 400
    assert "detail" in response.json()  # Should contain an error message for insufficient data

def test_predict_future_prices_with_valid_years():
    response = client.post("/predict-future-prices", json={
        "years": [2024, 2025],
        "data": [
            {"time_object": {"timestamp": "2019-06-01T00:00:00"},  "event_type": "sale", "attribute": {"price": 250000}},
            {"time_object": {"timestamp": "2020-06-01T00:00:00"},  "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01T00:00:00"},  "event_type": "sale", "attribute": {"price": 350000}},
        ]
    })
    assert response.status_code == 200
    result = response.json()
    predicted_prices = result["predicted_prices"]
    assert predicted_prices['2024']  # Prediction for 2024
    assert predicted_prices['2025']  # Prediction for 2025

# Test case 4: Missing price or timestamp
def test_missing_price_or_timestamp():
    # Missing price
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": [
            {"time_object": {"timestamp": "2020-06-01T00:00:00"}, "event_type": "sale", "attribute": {}},
            {"time_object": {"timestamp": "2021-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 350000}},
        ]
    })
    print(response.json())
    assert response.status_code == 400
    assert "detail" in response.json()  # Should contain an error message for missing price

    # Missing timestamp
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": [
            {"time_object": {},  "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01T00:00:00"},  "event_type": "sale", "attribute": {"price": 350000}},
        ]
    })
    assert response.status_code == 400
    assert "detail" in response.json()  # Should contain an error message for missing timestamp

# Test case 5: Invalid input format (non-list data)
def test_invalid_input_format():
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": "invalid_data"
    })
    assert response.status_code == 422
    assert "detail" in response.json()  # Should contain an error message for invalid input format

# Test case 6: Edge case - No data provided
def test_no_data():
    response = client.post("/predict-future-prices", json={
        "years": [2025],
        "data": []
    })
    assert response.status_code == 400
    assert "detail" in response.json()  # Should contain an error

def test_predict_missing_attribute():
    response = client.post("/predict", json={
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}],
        "x_attribute": "missing_x",
        "y_attribute": "missing_y",
        "x_values": [1, 2, 3]
    })
    print(response.json())
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

def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "microservice": "analytics","updated":"02/04/2025"}
