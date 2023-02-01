import tensorflow as tf
from tensorflow import keras
from utils import train_augs

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

class EventRecognizer(keras.Model):
    def __init__(self, **kwargs):
        self.base_model = InceptionV3(include_top=False)
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.output)

        self.aug_layer = train_augs

    def call(self, x):
        x = train_augs(x)
        x = self.model(x)
        return x

from glob import glob
datas = glob("./data/*")

x = EventRecognizer(datas[0])
print(x.shape)
    