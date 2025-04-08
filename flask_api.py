from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from PIL import Image

from cv.inference import preditct_from_flask_api

# Configuração de diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

app = Flask(__name__)
CORS(app)

# Configurar Flask para JSON
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json'


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    results = preditct_from_flask_api(file)[0].tolist()
    response = dict(zip(["calories", "mass", "fat", "carbs", "proteins"], results))
    return jsonify(response), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True, port=5000) 