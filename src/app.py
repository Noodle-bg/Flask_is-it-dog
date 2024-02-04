from flask import Flask, request
from fastai.vision.all import *
import pathlib
import json

app = Flask(__name__)

def is_cat(x):
    return x[0].isupper()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = PILImage.create(file)
    pred, idx, probs = learn.predict(img)
    result = {'Dog': float(probs[0]), 'Cat': float(probs[1])}
    return json.dumps(result)

if __name__ == '__main__':
    app.run(port=5000)