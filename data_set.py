import os
import cv2
import numpy as np

# Set the paths to the input image folders
input_folder = "/content/drive/MyDrive/Final Year Project/X4"
lr_folder = os.path.join(input_folder, 'LOW x4 URban100')
hr_folder = os.path.join(input_folder, 'HIGH x4 URban100')

# Set the desired size of the low-resolution images
lr_size = (256, 256)
hr_size = (256, 256)
# Load the images and preprocess them
train_x = []
train_y = []
test_x = []
test_y = []

for filename in os.listdir(lr_folder):
    if filename.endswith('.png'):
        # Load the low-resolution and high-resolution images
        lr_image = cv2.imread(os.path.join(lr_folder, filename)).astype('float64')
        hr_image = cv2.imread(os.path.join(hr_folder, filename.replace('_LR', '_HR'))).astype('float64')
        
        # Resize the low-resolution image to the desired size
        lr_image = cv2.resize(lr_image, lr_size, interpolation=cv2.INTER_CUBIC)
        hr_image = cv2.resize(hr_image, hr_size, interpolation=cv2.INTER_CUBIC)
        
        #if np.random.random() < 0.8:
        train_x.append(lr_image/255.0)
        train_y.append(hr_image/255.0)

# Convert the training and testing sets to NumPy arrays
train_x1 = np.array(train_x)
train_y1 = np.array(train_y)

# Split the dataset into training and testing sets (80-20 split)
train_x, train_y = train_x1[0:90,:,:,:], train_y1[0:90,:,:,:]
test_x, test_y = train_x1[80:100,:,:,:], train_y1[80:100,:,:,:]

def data_maker():
    return train_x, train_y, test_x, test_y