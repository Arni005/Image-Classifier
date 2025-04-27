import requests

# URL of the FastAPI application's predict endpoint
url = "http://127.0.0.1:8000/predict"

# Data to be sent in the POST request
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Make the POST request and capture the response
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Extract and print the prediction from the response JSON
    prediction = response.json()
    print(f"Predicted Species: {prediction['predicted_species']}")
else:
    # Print error if the request failed
    print(f"Error: Request failed with status code {response.status_code}")