import os
import json
from urllib.request import urlopen
import numpy as np


def _preprocess_time(str_time):
    str_time = str_time.split(" ")
    str_time = "_".join(str_time)
    return str_time


class APPAgent(object):
    root_url = "http://127.0.0.1:5000"

    @classmethod
    def predict(cls):
        options = "predict"
        url = os.path.join(cls.root_url, options)
        response = urlopen(url)
        return np.array(json.loads(response.read()))

    @classmethod
    def fit(cls, start, end=None, load_file_path=None, num_epochs=1):
        start = _preprocess_time(start)
        if end is None:
            end = "none"
        else:
            end = _preprocess_time(end)
        if load_file_path is None:
            load_file_path = "none"
        options = "fit/" + start + "/" + end + "/" + load_file_path + "/" + str(num_epochs)
        url = os.path.join(cls.root_url, options)
        response = urlopen(url)
        return response.read()

    @classmethod
    def update_data(cls):
        url = os.path.join(cls.root_url, "update_data")
        response = urlopen(url)
        return response.read()

    @classmethod
    def update_model(cls, num_epochs=1):
        url = os.path.join(cls.root_url, "update_model/" + str(num_epochs))
        response = urlopen(url)
        return response.read()
