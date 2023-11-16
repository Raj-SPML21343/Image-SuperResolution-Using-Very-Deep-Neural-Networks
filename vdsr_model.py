
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from data_set import data_maker





#from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, models
class PSNRCallback(Callback):
    def __init__(self, X_val, y_val):
        super(PSNRCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs=None):
        # Predict the validation set using the model
        y_pred = self.model.predict(self.X_val)
        # Calculate PSNR using NumPy
        psnr = tf.image.psnr(self.y_val, y_pred, max_val=2)
        # Add PSNR to the logs
        logs = logs or {}
        logs['val_psnr'] = np.mean(psnr)
        print("PSNR: {:.4f}".format(np.mean(psnr)))

# Define the VDSR model
def VDSR(input_shape=(None, None, 3)):
    model = models.Sequential()

    # Input layer
    model.add(layers.Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, padding='same', use_bias=False))
    model.add(layers.ReLU())

    # Features trunk blocks
    for _ in range(20):
        model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(layers.ReLU())

    # Output layer
    model.add(layers.Conv2D(3, kernel_size=(3, 3), padding='same', use_bias=False))

    # Initialize model weights
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0 / n))

    return model

# Compile the model
model_vdsr = VDSR()
model_vdsr.compile(loss='mse', optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

train_x, train_y, test_x, test_y = data_maker()
# Create the PSNR callback and fit the model
psnr_callback = PSNRCallback(test_x, test_y)
hist_vsdr = model_vdsr.fit(train_x,train_y, batch_size=16, epochs=100, validation_data=(test_x, test_y), callbacks=[psnr_callback])