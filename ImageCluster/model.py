import tensorflow as tf
from tensorflow import keras

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.mobilenet_v2 import MobileNetV2 
from keras.models import Model

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans

import pandas as pd

class RememberImages:
    """We gonna use pretrained model, MobileNet
    Because it should be light and fast! Also, MobileNet 
    is pretty efficient for our work. 
    if the result of MobileNet is not good, then next, we will use EfficientNet"""
    def __init__(self, img, ):
        self.img = img

    def _prep_img(self, ):
        preped_img = tf.image.resize(self.img, [224, 224])
        preped_img = tf.keras.applications.mobilenet.preprocess_input(preped_img*255)
        return preped_img
    
    def _create_model_return_features(self):
        base_model = MobileNetV2(weights="imagenet")
        base_model.layers.pop()
        model = Model(inputs=base_model.input, outputs=base_model.output) 
        print("🤖 Model Creating is Done! Shape...")
        preped_img = self._prep_img()
        features = model.predict(preped_img)
        print(features.shape)

        return features

    def _making_dict(self, features):
        df = pd.DataFrame()
        for feature in features:
            df

    def _clustering(self, features):
        df = pd.DataFrame()
        cls_model = DBSCAN()
        labels = cls_model.fit_predict(features)

        return labels
        
    def _viz(self, cls_labels):





        return 

