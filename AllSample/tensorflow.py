import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2.0, 3, 4])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1,2],
                             [3,4],
                             [5,6]], dtype=tf.float16)
print(rank_2_tensor)

rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor)

np.array(rank_2_tensor)
rank_2_tensor.numpy()

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])
print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

print(a + b, '\n')
print(a * b, '\n')
print(a @ b, '\n')

c = tf.constant([[4.0, 5], [10.0, 1]])

print(tf.reduce_max(c))
print(tf.argmax(c))
print(tf.nn.softmax(c))

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print('Type of every element:', rank_4_tensor.dtype)
print('Number of dimensions:', rank_4_tensor.ndim)
print('Shape of tensor:', rank_4_tensor.shape)
print('Elements along axis 0 of tensor:', rank_4_tensor.shape[0])
print('Elements along the last axis of tensor:', rank_4_tensor.shape[-1])
print('Total number of elements (3*3*4*5): ', tf.size(rank_4_tensor).numpy())

rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

print('First:', rank_1_tensor[0].numpy())
print('Second:', rank_1_tensor[1].numpy())
print('Last:', rank_1_tensor[-1].numpy())

print('Everything:', rank_1_tensor[:].numpy())
print('Before 4:', rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

print(rank_2_tensor.numpy())
print(rank_2_tensor[1, 1].numpy())
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

print(rank_3_tensor[:, :, 4])

var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)

print(var_x.shape.as_list())

reshaped = tf.reshape(var_x, [1, 3])

print(var_x.shape)
print(reshaped.shape)

print(rank_3_tensor)

print(tf.reshape(rank_3_tensor, [-1]))

print(tf.reshape(rank_3_tensor, [3*2, 5]), '\n')
print(tf.reshape(rank_3_tensor, [3, -1]))

try:
    tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
    print(f"{type(e).__name__}: {e}")

the_f64_tensor = tf.constant([2.2, 3.3, 4.8], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)