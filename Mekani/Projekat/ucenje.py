from __future__ import print_function

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

from sklearn import datasets
from sklearn.cluster import KMeans
import tensorflow as tf

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10 and h < 20 and h > 4 and w > 4:
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    
    return image_orig, sorted_regions
def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):

    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        test = matrix_to_vector(scale)
        ready_for_ann.append(test)
        
    return ready_for_ann

def convert_output(alphabet):

    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(53, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):

    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
 
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','O','P','Q','R','S','T','U','V','W','X','Y','Z']

img = load_image('C:/Users/Boris/Desktop/Mekani/Projekat/images/uci.jpg')
test = image_bin(image_gray(img))
img_roi, sorted_regions = select_roi(img.copy(),test)
plt.imshow(img_roi)
plt.show()

output = convert_output(alphabet)
inputa = prepare_for_ann(sorted_regions)
ann = create_ann()
ann = train_ann(ann, inputa, output)

results = ann.predict(np.array(inputa, np.float32))
print(display_result(results, alphabet))

model_json = ann.to_json()
with open("ann.json", "w") as json_file:
    json_file.write(model_json)
ann.save_weights("ann.h5")
print("Saved model to disk")