import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.utils import to_categorical
from tensorflow.keras.datasets import mnist
import random

model = tf.keras.models.load_model('mnist_model.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test.reshape(-1, 28 * 28)

index = random.randint(0, len(x_test))
random_image = x_test[index]
random_image = random_image.reshape(1, 28 * 28)

prediction = model.perdict(random_image)
prediction_label = prediction.argmax()
true_label = to_categorical(y_test, num_classes=10)[index].argmax()

plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"предсказанная цифра: {prediction_label}, настоящая цифра: {true_label} ")
plt.axis('off')
plt.show()