
# from flask import Flask, request, jsonify
# from fastai.basic_train import load_learner
# from fastai.vision import open_image
# from flask_cors import CORS,cross_origin
# app = Flask(__name__)
# CORS(app, support_credentials=True)

# app = Flask(__name__)

def is_cat(x):
    return x[0].isupper()

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# learn = load_learner('model.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     img = PILImage.create(file)
#     pred, idx, probs = learn.predict(img)
#     result = {'Dog': float(probs[0]), 'Cat': float(probs[1])}
#     return json.dumps(result)

# if __name__ == '__main__':
#     app.run(port=5000)
from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner(path='./src', file='model_dogs_and_cats.pkl')
classes = learn.data.classes


def predict_single(img_file):
    'function to take image and return prediction'
    prediction = learn.predict(open_image(img_file))
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()