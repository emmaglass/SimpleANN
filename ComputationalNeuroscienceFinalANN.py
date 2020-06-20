import numpy as np
import random as rand
import sklearn.model_selection as model_selection
from PIL import Image, ImageOps
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


def neural_network(num_arrays, array_size, rect_min, rect_max, rect_num, scale_factor):
	image_data = []
	box_data = []

	for array in range(num_arrays):
		for rect in range(rect_num):
			image= np.zeros(( array_size, array_size), dtype = np.int)
			box = np.zeros((1, 4), dtype = np.int)

			height_rect = rand.randint(rect_min, rect_max)
			wid_rect = rand.randint(rect_min, rect_max)
			x_coord_corner = rand.randint(0, array_size - wid_rect)
			y_coord_corner = rand.randint(0, array_size - height_rect)

			image[y_coord_corner:y_coord_corner+height_rect, x_coord_corner:
			                                    x_coord_corner+wid_rect] = 1
			image_data.append(image)
			box[rect] = [y_coord_corner, x_coord_corner, height_rect, wid_rect]
			box_data.append(box)

	image_data = np.asarray(image_data)
	image_data = image_data.reshape((image_data.shape[0], -1))
	box_data = np.asarray(box_data)
	box_data = box_data.reshape((box_data.shape[0], -1))

	X_train, X_test, Y_train, Y_test = model_selection.train_test_split
									   (image_data, box_data, train_size = .8, 
									   test_size = .2)
	
	model = Sequential([Dense(200, input_dim = array_size*array_size),
					    Activation('relu'), 
					    Dropout(0.2), 
					    Dense(4)])
	model.compile('adadelta', 'mse')

	model.fit(X_train, Y_train, epochs = 200, 
			  verbose = 2, 
			  validation_data = (X_test, Y_test))
	pred_y = model.predict(X_test).round(2)
	

	#increase array so 1000x1000 array
	Y_test = Y_test*100
	pred_y = pred_y*100

	actual_image_data = []
	predicted_image_data = []
	overlap_images = []

	for array in range(0,4):
		for rect in range(1):
			actual_image = np.zeros((scale_factor,scale_factor), dtype = np.int)
			predicted_image = np.zeros((scale_factor,scale_factor), dtype = np.int)

			actual_box = Y_test[array]
			predicted_box = pred_y[array]

			actual_height_box = actual_box[2]
			actual_wid_box = actual_box[3]
			actual_x_coord_corner = actual_box[1]
			actual_y_coord_corner =actual_box[0]

			predicted_height_box = int(predicted_box[2])
			predicted_wid_box = int(predicted_box[3])
			predicted_x_coord_corner = int(predicted_box[1])
			predicted_y_coord_corner = int(predicted_box[0])

			actual_image[actual_y_coord_corner:actual_y_coord_corner+actual_height_box, 
			             actual_x_coord_corner:actual_x_coord_corner+actual_wid_box] = 1
			predicted_image[predicted_y_coord_corner:predicted_y_coord_corner+predicted_height_box, 
			               predicted_x_coord_corner:predicted_x_coord_corner+predicted_wid_box] = 1

			actual_image_data.append(actual_image)
			predicted_image_data.append(predicted_image)

	actual_image_data = np.asarray(actual_image_data)
	actual_image_data_for_overalp = actual_image_data.reshape(
		                            (actual_image_data.shape[0], -1))

	predicted_image_data = np.asarray(predicted_image_data)
	predicted_image_data_for_overlap = predicted_image_data.reshape(
		                              (predicted_image_data.shape[0], -1))
			
	overlap_for_jpeg = actual_image_data + predicted_image_data

	overlap = actual_image_data_for_overalp + predicted_image_data_for_overlap

	image_no = 1
	for image in range(0,4):
		mat = np.reshape(overlap[image],(scale_factor, scale_factor))
		img = Image.fromarray(np.uint8(mat * 100) , 'L')
		im_invert = ImageOps.invert(img).save('trainingdata/image'+str(image_no)+'.png')
		image_no+=1

	intovuni = []
	for array in range(0,4):
		union = 0
		intersection = 0
		for i in overlap[array]:
			if i == 2:
				union+=1
				intersection+=1
			if i == 1:
				union+=1
		iou = intersection / union
		intovuni.append(iou)

	print(intovuni)



if __name__ == '__main__':
	neural_network(num_arrays = 1000, array_size = 10, rect_min = 1, rect_max = 5, rect_num = 1, scale_factor = 1000)




