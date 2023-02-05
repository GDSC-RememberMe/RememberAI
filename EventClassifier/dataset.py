import tensorflow as tf
from tensorflow import keras
from glob import glob
import pathlib

DATA_DIR = pathlib.Path(("D:/WIDER_v0.1/images/"))

## Define Configs
BATCH_SIZE = 32
SIZE = (256, 256)

def return_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=SIZE,
        batch_size=BATCH_SIZE
    )

    print(f"👀 Founded Classes...\n{val_ds.class_names}\n\n")

    BURFFER = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=BURFFER)
    val_ds = val_ds.cache().prefetch(buffer_size=BURFFER)


    normalizer = tf.keras.layers.Rescaling(1./255)

    norm_train = train_ds.map(lambda x, y: (normalizer(x), y))
    norm_val = val_ds.map(lambda x, y: (normalizer(x), y))

    return norm_train, norm_val