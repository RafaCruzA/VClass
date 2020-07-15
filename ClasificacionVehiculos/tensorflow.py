import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

input_km = np.array([1, 5, 8, 3, 2, 8, 15, 22, 10], dtype = float)
output_mts = np.array([1000, 5000, 3200, 8000, 15000, 22000, 10000], dtype = float)

for i, c in enumerate(input_km):
    print("{} kn = {} mts".format(c, output_mts[i]))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1, input_shape = [1])
])

model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

history = model.fit(input_km, output_mts, epochs = 10500, verbose = False)

import matplotlib.pyplot as plt
plt.xlabel("Interaction")
plt.ylabel("Magnitud del error(Loss)")
plt.plot(history.history['loss'])

print(model.predict([3.4]))

