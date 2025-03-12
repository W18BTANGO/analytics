# lambda_function.py
import json
import os

def lambda_handler(event, context):
    file_path = os.path.join(os.path.dirname(__file__), 'data.json')
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    print("Data:", data)
    return {
        'statusCode': 200,
        'body': json.dumps(data)
    }
