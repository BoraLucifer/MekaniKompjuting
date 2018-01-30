from __future__ import print_function

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections
import json
from pprint import pprint

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
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 20 and h < 20 and h > 4 and w > 4:
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2])
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

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
    ann.add(Dense(52, activation='sigmoid'))
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

def display_results(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]
'''
alphabet = ['a','a','a','a','a','b','b','b','b','b','c','c','c','c','c','d','d','d','d','d','e','e','e','e','e','f','f','f','f','f','g','g','g','g','g','h','h','h','h','h','i','i','i','i','i','j','j','j','j','j','k','k','k','k','k','l','l','l','l','l','m','m','m','m','m','n','n','n','n','n','o','o','o','o','o','p','p','p','p','p','q','q','q','q','q','r','r','r','r','r','s','s','s','s','s','t','t','t','t','t','u','u','u','u','u','v','v','v','v','v','x','x','x','x','x','y','y','y','y','y','z','z','z','z','z','A','A','A','A','A','B','B','B','B','B','C','C','C','C','C','D','D','D','D','D','E','E','E','E','E','F','F','F','F','F','G','G','G','G','G','H','H','H','H','H','I','I','I','I','I','J','J','J','J','J','K','K','K','K','K','L','L','L','L','L','M','M','M','M','M','N','N','N','N','N','O','O','O','O','O','P','P','P','P','P','Q','Q','Q','Q','R','R','R','R','S','S','S','S','S','T','T','T','T','T','U','U','U','U','U','V','V','V','V','V','W','W','W','W','W','X','X','X','X','X','Y','Y','Y','Y','Y','Z','Z','Z','Z','Z','w','w','w','w','w']
'''

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

json_file = open('ann.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ann = model_from_json(loaded_model_json)
ann.load_weights("ann.h5")
print("Loaded model from disk")

'''
text = input("Unesite broj karte koji zelite da proverite: ")
print(str(text))
print("mama mia")
text1 = "C:/Users/Boris/Desktop/Mekani/Projekat/images/" + text + ".jpg"
print(text1)
'''

test_color = load_image('C:/Users/Boris/Desktop/Mekani/Projekat/images/17.jpg')
test1 = test_color[15:43, 17:250]
test2 = image_bin(image_gray(test1))
plt.imshow(test2)
plt.show()

selected_test, test_numbers, distances = select_roi(test1.copy(),test2)
plt.imshow(selected_test)
plt.show()

distances = np.array(distances).reshape(len(distances), 1)

k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
k_means.fit(distances)

inputs = prepare_for_ann(test_numbers)
results = ann.predict(np.array(inputs, np.float32))
print(display_results(results, alphabet, k_means))
final = display_results(results, alphabet, k_means)

json_data = open("C:/Users/Boris/Desktop/Mekani/AllCards.json")
allCards = json.load(json_data)
save_key = ""
c=100
for key, value in allCards.items():
    k = levenshtein(final, key)    
    if (k < c):
        c=k
        save_key=key

print(save_key)
