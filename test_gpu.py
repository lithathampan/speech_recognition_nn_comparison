#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.test.is_gpu_available()
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
