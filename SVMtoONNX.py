import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier 

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a GradientBoostingClassifier with different hyperparameters
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the scikit-learn model
y_pred_sklearn = model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-learn Model Accuracy: {sklearn_accuracy * 100:.2f}%")

# Convert the model to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model
onnx_model_path = "gradient_boosting_breast_cancer.onnx"
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Load and test the ONNX model
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: X_test.astype(np.float32)})[0]

# Evaluate the ONNX model
onnx_accuracy = accuracy_score(y_test, onnx_pred)
print(f"ONNX Model Accuracy: {onnx_accuracy * 100:.2f}%")