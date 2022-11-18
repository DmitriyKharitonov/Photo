import os
from tensorflow import keras
import numpy as np
from PIL import ImageTk, Image, ImageOps
from variables import *


def main():
	model = keras.Sequential([
		keras.layers.Flatten(input_shape = (36,)),
		keras.layers.Dense(10, activation = 'relu'),
		keras.layers.Dense(6, activation = 'softmax')
	])

	model.compile(loss="categorical_crossentropy", 
          			optimizer="adam",
          			metrics = ["accuracy"])

	my_data = np.genfromtxt(f'./Датасет/data_0_random(2).csv', delimiter = ',')
	# my_data_test = np.genfromtxt('data_test.csv', delimiter = ',')
	# y_test, x_test = my_data_test[:, 0].astype('int'), my_data_test[:, 1:37].astype('int')
	y_train, x_train = my_data[:, 0].astype('int'), my_data[:, 1:37].astype('int')

	# x_test = x_test/255
	x_train = x_train/255
	# y_test_cat = keras.utils.to_categorical(y_test,6)
	y_train_cat = keras.utils.to_categorical(y_train,6)

	model.fit(x_train, y_train_cat, epochs = 5)
	model.save(f'./Модели/my_model_0_random(2).h5')
	# results = model.evaluate(x_test, y_test_cat, batch_size=64)
	# print('test loss, test acc:', results)


def test():
	my_data = np.genfromtxt('./Датасет/data — копия — копия.csv', delimiter = ',')
	x = my_data[1, 1:37].astype('int')
	x_ = np.expand_dims(x/255, axis = 0)
	model = keras.models.load_model('./Модели/my_model.h5')
	res = model.predict(x_)
	print(res)
	print(f'Распознанное направление: {np.argmax(res)}')
	

if __name__ == '__main__':
	main()
