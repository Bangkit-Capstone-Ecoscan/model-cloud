import os
import PIL
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd


# Assigning label names to the corresponding indexes
labels = {
    0: 'Bread',
    1: 'Dairy product',
    2: 'Dessert',
    3: 'Egg',
    4: 'Fried food',
    5: 'Meat',
    6: 'Noodles-Pasta',
    7: 'Rice',
    8: 'Seafood',
    9: 'Soup',
    10: 'Vegetable-Fruit'
}

# Load model
model = load_model('app/model/model_tf.h5')

# Load data carbon emission and food nutrition
carbon_csv = 'app/dataset/Carbon_Emission_of_Foods.csv'
nutrition_csv = 'app/dataset/food_nutrition.csv'

def predict_food(input_img):
    # Code to open the image
    img = PIL.Image.open(input_img)
    # Resizing the image to (256, 256)
    img = img.resize((256, 256))
    # Converting image to array
    img = np.asarray(img, dtype=np.float32)
    # Normalizing the image
    img = img / 255
    # Reshaping the image into a 4D array
    img = img.reshape(-1, 256, 256, 3)
    # Making prediction with the model
    predict = model.predict(img)
    # Getting the index corresponding to the highest value in the prediction
    predict = np.argmax(predict)

    # Read data from csv
    df = pd.read_csv(carbon_csv)
    df2 = pd.read_csv(nutrition_csv)

    # Get the predicted class from the model
    pred = labels[predict]

    # Filter the DataFrame based on the predicted class
    filter_carbon = df[df['id'] == pred.lower()]
    filter_nutrition = df2[df2['name'] == pred.lower()]

    # Extract the 'value' for the predicted class
    carbon = filter_carbon['value'].values[0]
    protein = filter_nutrition['protein'].values[0]
    calcium = filter_nutrition['calcium'].values[0]
    fat = filter_nutrition['fat'].values[0]
    carbohydrates = filter_nutrition['carbohydrates'].values[0]
    vitamins = filter_nutrition['vitamins'].values[0]

    # Return the results
    result = {
        'food-name': pred,
        'emission': f'{carbon:.2f} kg CO2',
        'protein': f'{protein:.2f} g',
        'calcium': f'{calcium:.2f} mg',
        'fat': f'{fat:.2f} g',
        'carbohydrates': f'{carbohydrates:.2f} g',
        'vitamins': vitamins
    }

    return result