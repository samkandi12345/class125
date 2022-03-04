import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784",version=1,return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scaled = X_train/255
X_test_scaled = X_test/255
clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)

def get_pred(image):
    im_pil = Image.open(image)
    bw = im_pil.convert("L")
    bw_resized = bw.resize((28,28),Image.ANTIALIAS)
    pixelfilter = 20
    minimum_pixel = np.percentile(bw_resized,pixelfilter)
    bw_resized_inverted_scale = np.clip(bw_resized-minimum_pixel,0,255)
    maximum_pixel = np.max(bw_resized)
    bw_resized_inverted_scale = np.asarray(bw_resized_inverted_scale)/maximum_pixel
    test_sample = np.array(bw_resized_inverted_scale).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

    

