from flask import request, jsonify
from app import app
from app.model_utils import predict_food

class PredictionResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@app.route("/predict/", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    image_path = "app/static/uploaded_images/temp_image.jpg"
    image.save(image_path)

    result = predict_food(image_path)
    response_json = PredictionResponse(**result).__dict__
    
    return jsonify(response_json)
