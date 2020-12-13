'''
module to create embedding for face

'''

import numpy as np
from keras.models import load_model


def get_embedding(model, face):
    "function to create face embedding"
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    return yhat[0]


data = np.load('dataset.npz')
train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

facenet_model = load_model('facenet_keras.h5')

# convert each face in the train set into embedding
emd_train_x = list()
for face in train_x:
    emd = get_embedding(facenet_model, face)
    emd_train_x.append(emd)
emd_train_x = np.asarray(emd_train_x)
print(emd_train_x.shape)

# convert each face in the test set into embedding
emd_test_x = list()
for face in test_x:
    emd = get_embedding(facenet_model, face)
    emd_test_x.append(emd)
emd_test_x = np.asarray(emd_test_x)
print(emd_test_x.shape)

# save arrays to one file in compressed format
np.savez_compressed('embeddings.npz', emd_train_x, train_y, emd_test_x, test_y)
