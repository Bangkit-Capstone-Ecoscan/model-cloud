from flask import request, jsonify
from app import app
from app.model_utils import predict_food
import requests
from io import BytesIO

class PredictionResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@app.route("/predict/", methods=["POST"])
def predict():
    # Get URL image from request
    image_url = request.json.get("image_url")
    print(image_url)

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Download image from URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400

    # Store image temporary
    image_path = "app/static/uploaded_images/temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(response.content)

    # Do prediction
    result = predict_food(image)
    response_json = PredictionResponse(**result).__dict__
    
    return jsonify(response_json)
