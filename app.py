from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Define a root route
@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}

# Define a /hello route
@app.get("/hello")
def goodbye_world():
    return {"message": "Goodbye, World!"}

# Define a /priya route
@app.get("/priya")
def hello_priya():
    return {"message": "hello, priya!"}

# Run the app with Uvicorn (FastAPI's recommended ASGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)