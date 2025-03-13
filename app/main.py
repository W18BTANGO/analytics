from fastapi import FastAPI

app = FastAPI(title="Analytics API", description="API for calculating analytics based on datasets", version="1.0.0")

@app.get("/value-growth")
async def value_growth(attribute: str, event_type: str):
    # Dummy calculation (replace)
    growth_rate = 5.2  
    return {"growth_rate": growth_rate}

@app.get("/predict")
async def predict(attribute: str, event_type: str, year: int):
    # Dummy calculation (replace)
    predicted_value = 123.45  
    return {"predicted_value": predicted_value}

@app.get("/highest")
async def highest(attribute: str, event_type: str):
    # Dummy calculation (replace)
    highest_value = 999  
    return {"highest_value": highest_value}

@app.get("/lowest")
async def lowest(attribute: str, event_type: str):
    # Dummy calculation (replace)
    lowest_value = 1  
    return {"lowest_value": lowest_value}

