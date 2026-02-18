import numpy as np
import pandas as pd

def activation(x: np.array) -> np.array:
    return np.tanh(2*(x-0.5)) / (2 * np.tanh(2 * 0.5)) + 0.5

def activation_derivative(x: np.array) -> np.array:
    return (1 - np.tanh(2*(x-0.5))**2) / np.tanh(2 * 0.5)

def loss(pred: np.array, target: np.array) -> float:
    assert pred.shape == target.shape
    return ((pred - target)**2).sum() / target.shape[0]

def loss_derivative(pred: np.array, target: np.array) -> np.array:
    assert pred.shape == target.shape
    return 2 * (pred - target)

def to_logic(pred: np.array) -> np.array:
    return (pred > 0.5).astype('i4')

def correct_percent(pred: np.array, target: np.array) -> float:
    return (to_logic(pred[:,0]) == to_logic(target[:,0])).mean() * 100

def predict_singular(model, features):
    return np.concat([model.forward(features[i:i+1]) for i in range(len(features))], axis=0)

def result_table_singular(model, features, targets):
    df = pd.DataFrame(features, columns=["in1", "in2"])
    df["pred_raw"] = predict_singular(model, features).squeeze()
    df["pred"] = to_logic(df["pred_raw"])
    df["target"] = to_logic(targets)
    return df

def result_table(model, features, targets):
    df = pd.DataFrame(features, columns=["in1", "in2"])
    df["pred_raw"] = model.forward(features).squeeze()
    df["pred"] = to_logic(df["pred_raw"])
    df["target"] = to_logic(targets)
    return df

def train_epoch(model, features, targets, lr):
    for i in range(features.shape[0]):
        model.zero_grad()
        pred = model.forward(features[i:i+1])
        model.backward(loss_derivative(pred, targets[i:i+1]))
        model.subtract_grad(lr)

def train_epoch_single(model, features, targets, lr):
    model.zero_grad()
    for i in range(features.shape[0]):
        pred = model.forward(features[i:i+1])
        model.backward(loss_derivative(pred, targets[i:i+1]))
    model.subtract_grad(lr)

class OneLayerSingular:
    def __init__(self):
        self.weights = np.random.normal(loc=0.5, scale=0.5, size=(2,1))
        self.bias = np.array([0.])
        self.a = None
        self.b = None
        self.weights_grad = np.zeros((2,1), dtype='f4')
        self.bias_grad = np.zeros((1,), dtype='f4')

    def forward(self, x: np.array) -> np.array:
        self.n = x.shape[0]
        assert x.shape == (self.n, 2)

        self.x = x[0,:]

        self.a = self.x @ self.weights + self.bias
        assert self.a.shape == (1,)

        self.b = activation(self.a)
        assert self.b.shape == (1,)

        return self.b[None,:]

    def zero_grad(self):
        self.weights_grad.fill(0)
        self.bias_grad.fill(0)
    
    def backward(self, grad_output: np.array):
        assert grad_output.shape == (self.n, 1)

        grad_output = grad_output[0,:]
        assert grad_output.shape == (1,)

        a_grad = grad_output * activation_derivative(self.a)
        assert a_grad.shape == (1,)

        weights_grad = np.outer(self.x, a_grad)
        assert weights_grad.shape == (2,1)

        bias_grad = a_grad
        assert bias_grad.shape == (1,)

        self.weights_grad += weights_grad
        self.bias_grad += bias_grad
    
    def subtract_grad(self, lr: float):
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad

class OneLayerBatch:
    def __init__(self):
        self.weights = np.random.normal(loc=0.5, scale=0.5, size=(2,1))
        self.bias = np.array([0.])
        self.a = None
        self.b = None
        self.n = None
        self.weights_grad = np.zeros((2,1), dtype='f4')
        self.bias_grad = np.zeros((1,), dtype='f4')

    def forward(self, x: np.array) -> np.array:
        self.x = x
        self.n = x.shape[0]
        assert x.shape == (self.n, 2)

        self.a = x @ self.weights + self.bias[None,:]
        assert self.a.shape == (self.n, 1)

        self.b = activation(self.a)
        assert self.b.shape == (self.n, 1)

        return self.b
    
    def zero_grad(self):
        self.weights_grad.fill(0)
        self.bias_grad.fill(0)

    def backward(self, grad_output: np.array):
        assert grad_output.shape == (self.n, 1)

        a_grad = grad_output * activation_derivative(self.a)
        assert a_grad.shape == (self.n, 1)

        weights_grad = self.x.T @ a_grad
        assert weights_grad.shape == (2,1)

        bias_grad = a_grad.sum(axis=0)
        assert bias_grad.shape == (1,)

        self.weights_grad += weights_grad
        self.bias_grad += bias_grad

    def subtract_grad(self, lr: float):
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad

class TwoLayerBatch:
    def __init__(self):
        self.weights1 = np.random.normal(loc=0.5, scale=0.5, size=(2,2))
        self.bias1 = np.array([0., 0.])
        self.weights2 = np.random.normal(loc=0.5, scale=0.5, size=(2,1))
        self.bias2 = np.array([0.])

        self.weights1_grad = np.zeros((2,2), dtype='f4')
        self.bias1_grad = np.zeros((2,), dtype='f4')
        self.weights2_grad = np.zeros((2,1), dtype='f4')
        self.bias2_grad = np.zeros((1,), dtype='f4')

        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.n = None

    def forward(self, x: np.array) -> np.array:
        self.x = x
        self.n = x.shape[0]
        assert x.shape == (self.n, 2)

        self.a = x @ self.weights1 + self.bias1[None,:]
        assert self.a.shape == (self.n, 2)
        self.b = activation(self.a)
        assert self.b.shape == (self.n, 2)

        self.c = self.b @ self.weights2 + self.bias2[None,:]
        assert self.c.shape == (self.n, 1)
        self.d = activation(self.c)
        assert self.d.shape == (self.n, 1)

        return self.d
    
    def zero_grad(self):
        self.weights1_grad.fill(0)
        self.bias1_grad.fill(0)
        self.weights2_grad.fill(0)
        self.bias2_grad.fill(0)

    def backward(self, grad_output: np.array):
        assert grad_output.shape == (self.n, 1)

        c_grad = grad_output * activation_derivative(self.c)
        assert c_grad.shape == (self.n, 1)


        b_grad = c_grad @ self.weights2.T
        assert b_grad.shape == (self.n, 2)
        a_grad = b_grad * activation_derivative(self.a)
        assert a_grad.shape == (self.n, 2)

        weights2_grad = self.b.T @ c_grad
        assert weights2_grad.shape == (2,1)
        bias2_grad = c_grad.sum(axis=0)
        assert bias2_grad.shape == (1,)

        weights1_grad = self.x.T @ a_grad
        assert weights1_grad.shape == (2,2)
        bias1_grad = a_grad.sum(axis=0)
        assert bias1_grad.shape == (2,)

        self.weights1_grad += weights1_grad
        self.bias1_grad += bias1_grad
        self.weights2_grad += weights2_grad
        self.bias2_grad += bias2_grad

    def subtract_grad(self, lr: float):
        self.weights1 -= lr * self.weights1_grad
        self.bias1 -= lr * self.bias1_grad
        self.weights2 -= lr * self.weights2_grad
        self.bias2 -= lr * self.bias2_grad
