from skimage.color import rgb2gray
import tensorflow as tf
import numpy as np

from .utils import resize_data


class DefaultProcessor(object):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def preprocess(self, observation, action, reward, terminal):
        return observation, action, reward, terminal

    def tensor_process(self, x):
        return x

    def get_input_shape(self):
        return self.input_shape


class AtariProcessor(DefaultProcessor):
    def __init__(self, height, width, reward_min=-1, reward_max=1):
        self.height = height
        self.width = width
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.input_shape = (height, width)
        self.scale = 255

    def preprocess(self, observation, action, reward, terminal):
        observation = resize_data([observation], self.height, self.width)[0]
        observation = rgb2gray(observation)
        observation = np.uint8(observation * self.scale)
        # Make the same rewards for every games
        reward = min(self.reward_max, max(self.reward_min, reward))
        return observation, action, reward, terminal

    def tensor_process(self, x):
        # change to (batch, width, hight, window_length)
        return tf.transpose(x, [0, 2, 3, 1]) / self.scale


class TradeProcessor(DefaultProcessor):

    def preprocess(self, observation, action, reward, terminal):
        reward = np.log(1. + reward)
        return observation, action, reward, terminal
