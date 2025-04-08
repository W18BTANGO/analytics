from fastapi.testclient import TestClient
from app.main import app

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
    assert len(response.json()["prediction"]) == 1  # Ensure prediction is returned for x_values


def test_average_by_attribute():
    response = client.post("/average-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["average_values"]["Downtown"] == 550000


def test_median_by_attribute():
    response = client.post("/median-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 700000}},
            {"time_object": {"timestamp": "2023-08-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    assert response.json()["median_values"]["Downtown"] == 600000


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


def test_count_by_time():
    response = client.post("/count-by-time", json={
        "time_format": "year",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {}},
            {"time_object": {"timestamp": "2023-08-15"}, "event_type": "sale", "attribute": {}},
            {"time_object": {"timestamp": "2022-05-20"}, "event_type": "sale", "attribute": {}}
        ]
    })
    print(response.json())
    assert response.status_code == 200
    assert response.json()["counts_by_time"]["2023"] == 2
    assert response.json()["counts_by_time"]["2022"] == 1


def test_outliers():
    response = client.post("/outliers", json={
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 100000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 200000}},
            {"time_object": {"timestamp": "2023-08-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 5000000}},
            {"time_object": {"timestamp": "2023-09-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 250000}}
        ]
    })
    assert response.status_code == 200
    assert 5000000 in response.json()["outliers"]


def test_min_max_by_attribute():
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 500000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Rhodes", "price": 300000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Rhodes", "price": 350000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Darlinghurst", "price": 900000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Darlinghurst", "price": 1000000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert result['maximum_attribute'] == 'Darlinghurst'
    assert result['minimum_attribute'] == 'Rhodes'

# Test case 2: Only one group with multiple values


def test_single_group():
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert result['maximum_attribute'] == 'Balmain'
    assert result['minimum_attribute'] == 'Balmain'

# Test case 3: Data with missing group or value attributes


def test_missing_group_or_value():
    # Missing group attribute
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 500000}},
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Balmain", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert result['maximum_attribute'] == 'Balmain'
    assert result['minimum_attribute'] == 'Balmain'

    # Missing value attribute
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown"}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert result['maximum_attribute'] == 'Downtown'
    assert result['minimum_attribute'] == 'Downtown'

# Test case 4: Multiple groups with equal average values


def test_equal_average_values():
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
            {"time_object": {"timestamp": "2023-07-01T00:00:00"}, "event_type": "sale", "attribute": {"suburb": "Uptown", "price": 500000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    # One of them will be chosen as max/min (implementation dependent)
    assert result['maximum_attribute'] in ['Downtown', 'Uptown']
    assert result['minimum_attribute'] in ['Downtown', 'Uptown']

# Test case 5: Edge case - No data


def test_no_data():
    response = client.post("/min-max-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": []
    })
    assert response.status_code == 400  # Expecting a 400 error for no data
    assert "detail" in response.json()  # Should contain error message

# Test case 6: Handling error (e.g., invalid input format)


def test_invalid_input_format():
    response = client.post("/min-max-by-attribute", json="invalid_data")
    assert response.status_code == 422
    assert "detail" in response.json()  # Should contain error message


def test_predict_future_values():
    response = client.post("/predict-future-values", json={
        "time_points": [2025, 2026, 2027],
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2020-06-01"}, "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01"}, "event_type": "sale", "attribute": {"price": 350000}},
            {"time_object": {"timestamp": "2022-06-01"}, "event_type": "sale", "attribute": {"price": 400000}},
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"price": 450000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert "predicted_values" in result
    predicted_values = result["predicted_values"]
    assert len(predicted_values) == 3  # Should return predictions for 3 future years
    assert all(year in predicted_values for year in ['2025', '2026', '2027'])

# Test case 2: Insufficient data (less than 2 data points)


def test_insufficient_data():
    response = client.post("/predict-future-values", json={
        "time_points": [2025],
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2020-06-01T00:00:00"}, "event_type": "sale", "attribute": {"price": 300000}}
        ]
    })
    assert response.status_code == 400
    assert "detail" in response.json()  # Should contain an error message for insufficient data


def test_predict_future_values_with_valid_years():
    response = client.post("/predict-future-values", json={
        "time_points": [2024, 2025],
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2020-06-01"}, "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01"}, "event_type": "sale", "attribute": {"price": 350000}},
            {"time_object": {"timestamp": "2022-06-01"}, "event_type": "sale", "attribute": {"price": 400000}}
        ]
    })
    assert response.status_code == 200
    result = response.json()
    assert "predicted_values" in result
    predicted_values = result["predicted_values"]
    assert '2024' in predicted_values and '2025' in predicted_values

# Test case 3: Missing data


def test_missing_price_or_timestamp():
    # Missing value attribute
    response = client.post("/predict-future-values", json={
        "time_points": [2025],
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2020-06-01"}, "event_type": "sale", "attribute": {}},
            {"time_object": {"timestamp": "2021-06-01"}, "event_type": "sale", "attribute": {"price": 350000}}
        ]
    })
    assert response.status_code == 400
    assert "detail" in response.json()

    # Missing timestamp
    response = client.post("/predict-future-values", json={
        "time_points": [2025],
        "value_attribute": "price",
        "data": [
            {"time_object": {}, "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01"}, "event_type": "sale", "attribute": {"price": 350000}}
        ]
    })
    assert response.status_code == 400
    assert "detail" in response.json()

# Test case 4: Empty data array


def test_empty_data_array():
    response = client.post("/predict-future-values", json={
        "time_points": [2025],
        "value_attribute": "price",
        "data": []
    })
    assert response.status_code == 400
    assert "detail" in response.json()

# Test case 5: No time_points


def test_no_years():
    response = client.post("/predict-future-values", json={
        "time_points": [],
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2020-06-01"}, "event_type": "sale", "attribute": {"price": 300000}},
            {"time_object": {"timestamp": "2021-06-01"}, "event_type": "sale", "attribute": {"price": 350000}}
        ]
    })
    assert response.status_code == 200
    assert "predicted_values" in response.json()
    assert response.json()["predicted_values"] == {}


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
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": [{"time_object": {}, "event_type": "sale", "attribute": {}}]
    })
    assert response.status_code == 400
    assert "No valid data found" in response.json()["detail"]


def test_median_by_attribute_no_valid_data():
    response = client.post("/median-by-attribute", json={
        "group_by_attribute": "suburb",
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
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": []
    })
    assert response.status_code == 400


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "microservice": "analytics", "updated": "02/04/2025"}


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
    outliers_response = client.post("/outliers", json={
        "value_attribute": "price",
        "data": [
            {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"price": predictions[0]}},
            {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"price": predictions[1]}},
            {"time_object": {"timestamp": "2023-08-01"}, "event_type": "sale", "attribute": {"price": 5000000}},  # Add more data points
            {"time_object": {"timestamp": "2023-09-01"}, "event_type": "sale", "attribute": {"price": 250000}}
        ]
    })
    assert outliers_response.status_code == 200
    assert "outliers" in outliers_response.json()


def test_workflow_sales_and_median_price():
    # Step 1: Record sales data
    sales_data = [
        {"time_object": {"timestamp": "2023-06-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 500000}},
        {"time_object": {"timestamp": "2023-07-01"}, "event_type": "sale", "attribute": {"suburb": "Downtown", "price": 600000}}
    ]

    # Count sales by year
    count_response = client.post("/count-by-time", json={
        "time_format": "year",
        "data": sales_data
    })
    assert count_response.status_code == 200
    assert count_response.json()["counts_by_time"]["2023"] == 2

    # Step 2: Calculate median price for the suburb
    median_response = client.post("/median-by-attribute", json={
        "group_by_attribute": "suburb",
        "value_attribute": "price",
        "data": sales_data
    })
    assert median_response.status_code == 200
    assert median_response.json()["median_values"]["Downtown"] == 550000
