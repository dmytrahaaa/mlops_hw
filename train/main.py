from fastai.vision import *
from flask import Flask, jsonify, request
from fastai.metrics import error_rate
import os
import mlflow
import mlflow.fastai
import mlflow.tracking

app = Flask(__name__)


@app.route('/ping')
def ping():
    return {'success': 200}


@app.route('/train')
def main():
    params = {}
    bs = 64
    path_img = r'./images'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'([^/]+)_\d+.jpg$'

    data = ImageDataBunch.from_name_re(
        path_img, fnames, pat, valid_pct=0.2, size=224, bs=bs, no_check=True
    ).normalize(imagenet_stats)

    mlflow.set_experiment("pets-ic-experiment")

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.load('./models/stage-1', strict=False, remove_module=True)

    mlflow.fastai.autolog()

    with mlflow.start_run() as run:
        learn.fit_one_cycle(4)
    mlflow.fastai.log_model(learn, 'model')
    mlflow.log_params(params)

    learn.export('./models/export.pkl')
    return {"success": 100}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
