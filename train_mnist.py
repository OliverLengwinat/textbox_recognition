# source: https://github.com/opensourcesblog/tensorflow-mnist/blob/master/mnist.py

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import input_data
import cv2
import numpy as np
import math
from scipy import ndimage

import mnist_preprocessing


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def train_and_predict(input_images):
	# create a MNIST_data folder with the MNIST dataset if necessary
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	"""
	a placeholder for our image data:
	None stands for an unspecified number of images
	784 = 28*28 pixel
	"""
	x = tf.placeholder("float", [None, 784])

	# we need our weights for our neural net
	W = tf.Variable(tf.zeros([784,10]))
	# and the biases
	b = tf.Variable(tf.zeros([10]))

	"""
	softmax provides a probability based output
	we need to multiply the image values x and the weights
	and add the biases
	(the normal procedure, explained in previous articles)
	"""
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	"""
	y_ will be filled with the real values
	which we want to train (digits 0-9)
	for an undefined number of images
	"""
	y_ = tf.placeholder("float", [None,10])

	"""
	we use the cross_entropy function
	which we want to minimize to improve our model
	"""
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))

	"""
	use a learning rate of 0.01
	to minimize the cross_entropy error
	"""
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	# initialize all variables
	init = tf.initialize_all_variables()

	# create a session
	sess = tf.Session()
	sess.run(init)

	# use 1000 batches with a size of 100 each to train our net
	for i in range(1000):
	  batch_xs, batch_ys = mnist.train.next_batch(100)
	  # run the train_step function with the given image values (x) and the real output (y_)
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	"""
	Let's get the accuracy of our model:
	our model is correct if the index with the highest y value
	is the same as in the real digit vector
	The mean of the correct_prediction gives us the accuracy.
	We need to run the accuracy function
	with our test set (mnist.test)
	We use the keys "images" and "labels" for x and y_
	"""
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	# create an an array where we can store our 4->1 pictures
	images = np.zeros((len(input_images),784))
	# and the correct values
	correct_vals = np.zeros((len(input_images),10))

	# we want to test our images which you saw at the top of this page
	i = 0
	# for no in [8,0,4,3]:
	for image in input_images:
		"""
		we need to store the flatten image and generate
		the correct_vals array
		correct_val for the first digit (9) would be
		[0,0,0,0,0,0,0,0,0,1]
		"""
		current_image = cv2.imread("output_digits/"+image)
		images[i] = mnist_preprocessing.flatten(cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY))
		cv2.waitKey()
		correct_val = np.zeros((10))
		correct_val[9] = 1	# set correct value to 9 for now
		correct_vals[i] = correct_val
		i += 1

	"""
	the prediction will be an array with four values,
	which show the predicted number
	"""
	prediction = tf.argmax(y,1)
	"""
	we want to run the prediction and the accuracy function
	using our generated arrays (images and correct_vals)
	"""
	print(sess.run(prediction, feed_dict={x: images, y_: correct_vals}))
	#return(sess.run(prediction, feed_dict={x: images, y_: correct_vals}))
	#print(sess.run(accuracy, feed_dict={x: images, y_: correct_vals}))
