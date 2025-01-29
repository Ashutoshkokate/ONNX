import tensorflow as tf
import numpy as np  # Import numpy here
import tf2onnx
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score

# Load the pre-trained MobileNetV2 model 
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add a classification layer for MNIST (10 classes)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape and normalize data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Repeat grayscale channel to create 3 channels for MobileNetV2
x_train = np.repeat(x_train, 3, axis=3)
x_test = np.repeat(x_test, 3, axis=3)

# Resize images to 32x32 using tf.image.resize
x_train = tf.image.resize(x_train, [32, 32]).numpy()  
x_test = tf.image.resize(x_test, [32, 32]).numpy()   

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"TensorFlow Model Accuracy on MNIST: {accuracy * 100:.2f}%")

# Convert the model to ONNX format
input_signature = [tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32)]  # Define the input signature
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save the ONNX model
onnx.save(onnx_model, "mnist_mobilenetv2.onnx")
print('TensorFlow model converted to ONNX and saved as mnist_mobilenetv2.onnx')

# Load the ONNX model using onnxruntime
ort_session = ort.InferenceSession("mnist_mobilenetv2.onnx")

# Get the input name from the ONNX model
input_name = ort_session.get_inputs()[0].name

# Run inference using the ONNX model
onnx_preds = ort_session.run(None, {input_name: x_test.astype(np.float32)})[0]
onnx_pred_labels = np.argmax(onnx_preds, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate the ONNX model's accuracy
onnx_accuracy = accuracy_score(y_test_labels, onnx_pred_labels)
print(f"ONNX Model Accuracy on MNIST: {onnx_accuracy * 100:.2f}%")
