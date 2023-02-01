import tensorflow as tf


train_augs = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomCrop(200, 200),
    tf.keras.layers.RandomBrightness(0.3),
])
print("ccc")
