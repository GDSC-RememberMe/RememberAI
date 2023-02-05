import tensorflow as tf
from tensorflow import keras
from utils import train_augs

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

class EventRecognizer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(EventRecognizer, self).__init__()
        self.base_model = InceptionV3(include_top=False, input_shape=(256, 256, 3))
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        self.aug_layer = train_augs
        self.fc = tf.keras.layers.Dense(19)
        
        self.build(input_shape=(256,256,3))

    def call(self, x):
        x = self.aug_layer(x)
        x = self.model(x)
        output = self.fc(x)
        return output
