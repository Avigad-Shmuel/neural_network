import numpy as np

# === Activation functions ===
def softmax(x):
	exps = np.exp(x - np.max(x))
	return exps / np.sum(exps)

def sigmoid(x: float) -> float:
	return 1 / (1 + np.exp(-x))

def sigmoid_der_from_output(s):
	return s * (1 - s)

def relu(x):
	return np.maximum(0, x)

def relu_der(x):
	return (x > 0).astype(float)

# === Layer class ===
class Layer:
	def __init__(self, n: int, m: int, activation='relu'):
		if activation == 'relu':
			self.W = np.random.randn(m, n) * np.sqrt(2.0 / n)
			self.bias = np.ones(m) * 0.1
		else:
			# Xavier for sigmoid and softmax
			self.W = np.random.randn(m, n) * np.sqrt(1.0 / n)
			self.bias = np.zeros(m)

		self.activation = activation
		self.z = None
		self.a = None
		self.W_grad = np.zeros((m, n))
		self.b_grad = np.zeros(m)
		self.acc_W_grad = np.zeros((m, n))
		self.acc_b_grad = np.zeros(m)

	def forward(self, x):
		self.z = self.W @ x + self.bias
		if self.activation == 'relu':
			self.a = relu(self.z)
		elif self.activation == 'sigmoid':
			self.a = sigmoid(self.z)
		elif self.activation == 'softmax':
			self.a = softmax(self.z)
		return self.a.copy()

# === Neural network ===
class NeuralNetwork:
	def __init__(self, layers: list[Layer]):
		self.layers: list[Layer] = layers

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		for layer in self.layers:
			inputs = layer.forward(inputs)
		return inputs

	def cross_entropy(self, predicted: np.ndarray, target: np.ndarray) -> float:
		return -np.sum(target * np.log(predicted + 1e-9))  # avoid log(0)

	def loss(self, predicted: np.ndarray, target: np.ndarray) -> float:
		if self.layers[-1].activation == 'softmax':
			return self.cross_entropy(predicted, target)
		else:
			return np.sum(((predicted - target) ** 2)/2)

	def one_example(self, inputs, y):
		y_hat = self.forward(inputs)
		loss = self.loss(y_hat, y)
		layer = self.layers[-1]
		a_grad = layer.a - y

		# Backward pass
		for i in range(len(self.layers) - 1, 0, -1):
			layer = self.layers[i]
			if layer.activation == 'relu':
				z_grad = a_grad * relu_der(layer.z)
			elif layer.activation == 'sigmoid':
				z_grad = a_grad * sigmoid_der_from_output(layer.a)
			elif layer.activation == 'softmax':
				z_grad = a_grad  # already dL/dz for softmax+CE

			layer.acc_b_grad += z_grad
			layer.W_grad = np.outer(z_grad, self.layers[i - 1].a)
			layer.acc_W_grad += layer.W_grad
			a_grad = layer.W.T @ z_grad

		# First layer
		layer = self.layers[0]
		if layer.activation == 'relu':
			z_grad = a_grad * relu_der(layer.z)
		elif layer.activation == 'sigmoid':
			z_grad = a_grad * sigmoid_der_from_output(layer.a)
		elif layer.activation == 'softmax':
			z_grad = a_grad
		layer.acc_b_grad += z_grad
		layer.W_grad = np.outer(z_grad, inputs)
		layer.acc_W_grad += layer.W_grad

		return loss

	def apply_gradients(self, lr: float):
		for layer in self.layers:
			layer.W -= lr * layer.acc_W_grad
			layer.bias -= lr * layer.acc_b_grad
			layer.acc_W_grad[:] = 0
			layer.acc_b_grad[:] = 0

	def predict(self, x):
		output = self.forward(x)
		return np.argmax(output)
