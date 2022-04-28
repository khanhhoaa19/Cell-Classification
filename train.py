from PIL import Image
from numpy import asarray
import os

#0. Breast invasive carcinoma
#1. Kidney renal clear cell carcinoma
#2. Kidney renal papillary cell carcinoma
#3. Lung adenocarcinoma
#4. Lung squamous cell carcinoma
#5. Prostate adenocarcinoma

def change_to_ndarray(path, data_train):
    img = Image.open(path)
    numpydata = asarray(img)
    data_train.append(numpydata)

def change_to_ndarray_test(path, data, label):
    img = Image.open(path)
    numpydata = asarray(img)
    data.append(numpydata)
    label.append(5)

path = "D:/dataset/32x32/"
path_validatioon = "D:/dataset/32x32/validation/"
path_test = "D:/dataset/32x32/test/"
X_train = []
y_train = []
X_validation = []
y_validation = []
X_test = []
y_test = []
for folder in os.listdir(path):
    if folder == "Breast invasive carcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(2148):
            y_train.append(0)
    if folder == "Kidney renal clear cell carcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(1206):
            y_train.append(1)
    if folder == "Kidney renal papillary cell carcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(4365):
            y_train.append(2)
    if folder == "Lung adenocarcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(1330):
            y_train.append(3)
    if folder == "Lung squamous cell carcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(1233):
            y_train.append(4)
    if folder == "Prostate adenocarcinoma":
        for img in os.listdir(path + folder):
            file = path + folder + "/" + img
            change_to_ndarray(file, X_train)
        for i in range(2271):
            y_train.append(5)

for folder in os.listdir(path_validatioon):
    if folder == "Breast invasive carcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(338):
            y_validation.append(0)
    if folder == "Kidney renal clear cell carcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(338):
            y_validation.append(1)
    if folder == "Kidney renal papillary cell carcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(1039):
            y_validation.append(2)
    if folder == "Lung adenocarcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(397):
            y_validation.append(3)
    if folder == "Lung squamous cell carcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(445):
            y_validation.append(4)
    if folder == "Prostate adenocarcinoma":
        for img in os.listdir(path_validatioon + folder):
            file = path_validatioon + folder + "/" + img
            change_to_ndarray(file, X_validation)
        for i in range(275):
            y_validation.append(5)

for img in os.listdir(path_test):
    file = path_test + img
    change_to_ndarray_test(file, X_test, y_test)


print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 1000
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 6.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 6), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(6))

    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 6)

rate = 0.004

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training LeNet on dataset, please wait\n")

    train_accuracy_list = []
    validation_accuracy_list = []

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = evaluate(X_train, y_train)
        train_accuracy_list.append(train_accuracy)

        validation_accuracy = evaluate(X_validation, y_validation)
        validation_accuracy_list.append(validation_accuracy)

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print('Validation Accuracy = {:.3f}'.format(validation_accuracy))
        print('----------------------------')
    saver.save(sess, './lenet')
    print("Model saved")

import matplotlib.pyplot as plt
plt.plot(train_accuracy_list)
plt.title("Training Accuracy")
plt.show()

plt.plot(validation_accuracy_list)
plt.title("Validation Accuracy")
plt.show()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate(X_train, y_train)
    print("Training Accuracy = {:.3f}".format(train_accuracy))

    valid_accuracy = evaluate(X_validation, y_validation)
    print("Validation Accuracy = {:.3f}".format(valid_accuracy))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
