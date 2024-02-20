import tensorflow as tf
import numpy as np

# define model architecture
model = tf.keras.Sequential([tf.keras.layers.Dense(units = 1, input_shape = [1])])
# Compile the model
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
#  define training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype= float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype= float)
# train the model
model.fit(xs, ys, epochs=500)
# Make prediction
# print(model.predict([10.0]))  # output == 19

import tensorflow as tf

# tf.print("hello world")

# # Create two constant tensors
# a = tf.constant(3)
# b = tf.constant(4)

# # Add them together
# result = tf.add(a, b)

# # Create a TensorFlow session
# with tf.session() as sess:
# # Run the operation and print the result
#     output = sess.run(result)
# print("TensorFlow is working! Result:", output)

# import express from 'express';
# import APIToolkit from 'apitoolkit-express';

# const app = express();
# const port = 3000;

# const apitoolkitClient = await APIToolkit.NewClient({ apiKey: '<API-KEY>' });
# app.use(apitoolkitClient.expressMiddleware);

# app.get('/', (req, res) => {
#    res.send('Hello World!');
# });

# app.listen(port, () => {
#    console.log(`Example app listening on port ${port}`);
# });    