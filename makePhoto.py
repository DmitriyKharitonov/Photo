from tensorflow import keras
from PIL import ImageTk, Image, ImageOps, ImageDraw
import numpy as np
import os
from variables import *


def main():
	dir_ = './Фото_оригинал/'
	name = '683308'
	format_ = '.jpg'
	model_path = './Модели/'
	model_name = model_path + 'my_model_0_random.h5'
	path = dir_+name+format_
	background = (255,255,255)

	image = Image.open(path).convert('L')
	height, weidth = image.size
	model = keras.models.load_model(model_name)


	im = Image.new('RGB', (weidth, height), (background))
	draw = ImageDraw.Draw(im)

	tmp = np.asarray(image)
	current_x = 0
	current_y = 0

	while current_y < height - network_size:
		if (current_x >= weidth - network_size):
			current_x = 0
			current_y += step

		data = np.ravel(tmp[current_x:current_x+network_size, current_y:current_y+network_size])
		data = np.expand_dims(data/255, axis = 0)

		result = np.argmax(model(data))

		if result == 1:
			draw.line(
			    xy=(
			        (current_x+stroke_size/2, current_y),
			        (current_x+stroke_size/2, current_y+stroke_size),
			    ), fill='black')
		elif result == 2:
			draw.line(
			    xy=(
			        (current_x, current_y+stroke_size/2),
			        (current_x+stroke_size, current_y+stroke_size/2),
			    ), fill='black')
		elif result == 3:
			draw.line(
			    xy=(
			        (current_x, current_y),
			        (current_x+stroke_size, current_y+stroke_size),
			    ), fill='black')
		elif result == 4:
			draw.line(
			    xy=(
			        (current_x, current_y+stroke_size),
			        (current_x+stroke_size, current_y),
			    ), fill='black')

		current_x += 2

	im = im.rotate(270, expand=True)
	im = im.transpose(Image.FLIP_LEFT_RIGHT)
	im.save(f'./Фото_нейросеть/{name}_random.png')


if __name__=='__main__':
	main()