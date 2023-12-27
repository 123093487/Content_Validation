import tensorflow as tf
import numpy as np
import cv2

from keras.models import load_model
import tensorflow as tf
import numpy as np
model = load_model('97.5_per.h5')
image = cv2.imread('QbGNOmpOEKO2Hi6k62SDW9kGQvK-YJPHd2NDRpMG3Dc.png')
resize = tf.image.resize(image, (500,500))
np.expand_dims(resize, 0).shape
value = model.predict(np.expand_dims(resize/255 , 0))
if value>0.5:
    print('valid')
else:
    print('invalid')
