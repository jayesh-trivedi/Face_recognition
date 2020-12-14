"""
face recognition model using MTCNN and Facenet

"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from extract_face import extract_face
from create_embedding import get_embedding


# to counter OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# loading pre_classified dataset
data = np.load('dataset.npz')
train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# loading pre-trained model
facenet_model = load_model('facenet_keras.h5')
print('Loaded Model')

embedding = np.load('embeddings.npz')
emd_train_x, train_y, emdtest_x, test_y = embedding['arr_0'], embedding['arr_1'], embedding['arr_2'], embedding['arr_3']


print("Dataset: train=%d, test=%d" % (emd_train_x.shape[0], emdtest_x.shape[0]))
in_encoder = Normalizer()
emd_train_x_norm = in_encoder.transform(emd_train_x)
emd_test_x_norm = in_encoder.transform(emdtest_x)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(train_y)
train_y_enc = out_encoder.transform(train_y)
test_y_enc = out_encoder.transform(test_y)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(emd_train_x_norm, train_y_enc)

# predict
yhat_train = model.predict(emd_train_x_norm)
yhat_test = model.predict(emd_test_x_norm)

# score
score_train = accuracy_score(train_y_enc, yhat_train)
score_test = accuracy_score(test_y_enc, yhat_test)

# summarize
print('Accuracy:train = %.3f,test = %.3f' %
      (score_train * 100, score_test * 100))

# running custom test case
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.write('images/cap.jpg', frame)
cap.release()
face = extract_face('images/test.jpg')
face = np.asarray(face)
embedding_test = get_embedding(facenet_model, face)
embedding_test = embedding_test.reshape(1, -1)
embedding_norm = in_encoder.transform(embedding_test)
yhat_class = model.predict(embedding_norm)
yhat_prob = model.predict_proba(embedding_norm)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

plt.imshow(face)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
if class_probability < 60:
    plt.title('Cant identify')
plt.show()
