import numpy as np
import cv2
from keras.models import load_model
import time

model = load_model("3DCarEncoder.keras")

# for _ in range(10):
#     start = time.time()
#     model.predict(np.random.random((1, 96, 96, 1)), verbose=False)
#     print("Model process time:", time.time()-start)
#
# print('Batch size 100')
# for _ in range(10):
#     start = time.time()
#     model.predict(np.random.random((100, 96, 96, 1)), verbose=False)
#     print("Model process time:", time.time()-start)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def process_batch(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img[..., ::-1]
    img = rgb2gray(img)
    img = np.round(img / 100) * 50
    return img

def process_img(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img[..., ::-1]
    img = rgb2gray(img)
    img = np.round(img / 100) * 50
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(imgs):
    features = model.predict(imgs, verbose=False)
    return features
