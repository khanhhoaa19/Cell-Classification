from PIL import Image
from numpy import asarray
import os
import tensorflow as tf
from train import evaluate, saver
#0. Breast invasive carcinoma
#1. Kidney renal clear cell carcinoma
#2. Kidney renal papillary cell carcinoma
#3. Lung adenocarcinoma
#4. Lung squamous cell carcinoma
#5. Prostate adenocarcinoma

def change_to_ndarray_new(path, data, label):
    img = Image.open(path)
    numpydata = asarray(img)
    data.append(numpydata)
    label.append(5)

path_test = "D:/dataset/32x32/test/"
X_test = []
y_test = []

for img in os.listdir(path_test):
    file = path_test + img
    change_to_ndarray_new(file,X_test, y_test)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))