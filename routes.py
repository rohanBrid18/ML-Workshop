from flask import Blueprint, request, jsonify
import pickle

model = pickle.load(open('iris.model', 'rb'))

model_predictor = Blueprint('model_predictor', __name__)

flowers = ["Setosa", "Versicolor", "Virginica"]

@model_predictor.route('/predict', methods=['POST'])
def predict_value():
    sepal_w = request.get_json()['sepal_width']
    sepal_l = request.get_json()['sepal_length']
    petal_w = request.get_json()['petal_width']
    petal_l = request.get_json()['petal_length']

    p = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    print(p)

    return jsonify(
        {
            'flower': flowers[p[0]]
        }
    )