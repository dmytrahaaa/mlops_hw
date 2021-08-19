from fastai.vision import *
from fastai.metrics import error_rate
import time
import logging
import os
# import mlflow.fastai

logging.basicConfig(level=logging.INFO)


def main():

    bs = 64
    path_img = r'../images'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'([^/]+)_\d+.jpg$'


    # mlflow.fastai.autolog()

    data = ImageDataBunch.from_name_re(
        path_img, fnames, pat, valid_pct=0.2, size=224, bs=bs, no_check=True
                                    ).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    logging.info("loading latest parameters")

    learn.load('/home/autodoc/PycharmProjects/mlops_hw/app/models/stage-1', strict=False, remove_module=True)

    logging.info("start fit")

    # with mlflow.start_run() as run:
    #     learn.fit_one_cycle(4)
    #     mlflow.fastai.log_model(learn, "model")

    logging.info("end fit")

    # learn.save('/home/autodoc/PycharmProjects/mlops_hw/app/models/stage-2')
    learn.export('/home/autodoc/PycharmProjects/mlops_hw/app/models/export.pkl')


main()