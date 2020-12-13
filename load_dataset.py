'''
to load dataset for training

'''

import os
import numpy as np
from extract_face import extract_face

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_face(dataset_sub_path):
    " to extract individual faces "
    faces = list()
    files = [f for f in os.listdir(dataset_sub_path) if not f.startswith('.')]
    for filename in files:
        if not filename.startswith('.'):
            path = dir + filename
            face = extract_face(path)
            faces.append(face)
    return faces


def load_dataset(dataset_path):
    " list for faces and labels "
    inputs, outputs = list(), list()
    subdirectories = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
    for subdir in subdirectories:
        dataset_sub_path = dir + subdir + '/'
        faces = load_face(dataset_sub_path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
        inputs.extend(faces)
        outputs.extend(labels)
    return np.asarray(inputs), np.asarray(outputs)


train_x, train_y = load_dataset('dataset/data/train/')
print(train_x.shape, train_y.shape)
test_x, test_y = load_dataset('dataset/data/val/')
print(test_x.shape, test_y.shape)

# save and compress the dataset for further use
np.savez_compressed('dataset.npz', train_x, train_y, test_x, test_y)
