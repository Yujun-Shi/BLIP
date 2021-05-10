import cv2
import os

import pickle
import numpy as np

def get_total(data):
    data_x, data_y = [], []
    for k, v in data.items():
        for i in range(len(v)):
            data_x.append(v[i])
            data_y.append(k)
    d = {}
    d['images'] = data_x
    d['labels'] = data_y
    return d

# load all the data
data_path = './MI_raw'
all_folder_name = os.listdir(data_path)
data_dict = dict()
data_dict['images'] = []
data_dict['labels'] = []
for class_id,f in enumerate(all_folder_name):
    folder_path = os.path.join(data_path, f)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img_array = cv2.imread(img_path)
        data_dict['images'].append(img_array)
        data_dict['labels'].append(class_id)
    print(class_id)
data = data_dict['images']
labels = data_dict['labels']

# fix random seed as ACL (https://github.com/facebookresearch/Adversarial-Continual-Learning/blob/master/data/split_miniimagenet.py)
np.random.seed(1234)

# split and pickle
# we want 500 for training, 100 for test for each class
n = 500 

# split data into classes, 600 images per class
class_dict = {}
for i in range(len(set(labels))):
    class_dict[i] = []

for i in range(len(data)):
    class_dict[labels[i]].append(data[i])

# Split data for each class to 500 and 100
x_train, x_test = {}, {}
for i in range(len(set(labels))):
    np.random.shuffle(class_dict[i])
    x_test[i] = class_dict[i][n:]
    x_train[i] = class_dict[i][:n]

# mix the data
d_train = get_total(x_train)
d_test = get_total(x_test)

with open(os.path.join('train.pkl'), 'wb') as f:
    pickle.dump(d_train, f)
with open(os.path.join('test.pkl'), 'wb') as f:
    pickle.dump(d_test, f)
