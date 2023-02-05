import tensorflow as tf

from dataset import *
from model import *
from utils import *

norm_train, norm_val = return_dataset()

img, label =  next(iter(norm_train))
print(img.shape, label.shape)

model = EventRecognizer()
print(model(img).shape)

model.compile(optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
            )

hist = model.fit(norm_train, epochs=10, validation_data=norm_val)

print(model.summary())


