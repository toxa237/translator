import tensorflow as tf
from tensorflow.python.platform import build_info as build

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(f"Cuda Version: {build.build_info['cuda_version']}")
# tf.debugging.set_log_device_placement(True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: # Only if there is at least one GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("gpu not found, gpu init error!")

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)

    a = tf.random.normal((10, 10))
    b = tf.random.normal((10, 10))
    c = tf.matmul(a, b)
    print(c)
