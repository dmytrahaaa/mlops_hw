from pathlib import Path
from flask import Flask, jsonify, request
from fastai.vision import *
import os
import re

app = Flask(__name__)

path = Path('./models')
learner = load_learner(path, 'export.pkl')


@app.route('/ping')
def ping():
    return {'success': 200}


@app.route('/predict/<src_folder>')
def predict(src_folder):
    results = {}
    file_names = os.listdir(src_folder)

    for file_name in file_names:
        payload = request.args
        print(payload)
        path = os.path.join(src_folder, file_name)
        img = open_image(path)
        pred_class, pred_idx, outputs = learner.predict(img)
        proc = str(outputs[pred_idx])[9:11]
        new_folder = 'high_score' if int(proc) >= 70 else 'low_score'
        shutil.move(path, f'./{new_folder}')
        os.remove(file_name)
        result = f"category: '%s', accuracy: '%s'" % (pred_class, proc)
        results[file_name] = result
    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
