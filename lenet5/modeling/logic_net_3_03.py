import numpy as np
import pandas as pd
import lenet5.modeling.autograd as ag
from lenet5.modeling.autograd import Value

def activation(x):
    return ag.tanh(2*(x-0.5)) / (2 * ag.tanh(2 * 0.5)) + 0.5

class OneLayer:
    def __init__(self, generator: np.random.Generator = np.random.default_rng()):
        self.w1 = Value(generator.normal(loc=0.5, scale=0.5), name='w1')
        self.w2 = Value(generator.normal(loc=0.5, scale=0.5), name='w1')
        self.b = Value(.0, name='b')
        self._parameters = [self.w1, self.w2, self.b]

    def parameters(self):
        return self._parameters
        
    def forward(self, x1, x2):
        return activation(self.w1 * x1 + self.w2 * x2 + self.b)

class TwoLayer:
    def __init__(self, rand = None):
        if rand is None:
            rand = lambda: np.random.normal(loc=0.5, scale=0.5)
        self.w111 = Value(rand(), name='w111')
        self.w112 = Value(rand(), name='w112')
        self.w121 = Value(rand(), name='w121')
        self.w122 = Value(rand(), name='w122')
        self.b11 = Value(.0, name='b11')
        self.b12 = Value(.0, name='b12')

        self.w21 = Value(rand(), name='w21')
        self.w22 = Value(rand(), name='w22')
        self.b2 = Value(.0, name='b2')

        self._parameters = [self.w111, self.w112, self.w121, self.w122, self.b11, self.b12, self.w21, self.w22, self.b2]

    def parameters(self):
        return self._parameters
    
    def forward(self, x1, x2):
        h1 = activation(self.w111 * x1 + self.w121 * x2 + self.b11)
        h2 = activation(self.w112 * x1 + self.w122 * x2 + self.b12)
        return activation(self.w21 * h1 + self.w22 * h2 + self.b2)

class MiniTwoLayer:
    def __init__(self, generator: np.random.Generator = np.random.default_rng()):
        self.w0 = Value(generator.normal(loc=0.5, scale=0.5), name='w0')
        self.w1 = Value(generator.normal(loc=0.5, scale=0.5), name='w1')
        self.w2 = Value(generator.normal(loc=0.5, scale=0.5), name='w2')

        self.u0 = Value(generator.normal(loc=0.5, scale=0.5), name='u0')
        self.u1 = Value(generator.normal(loc=0.5, scale=0.5), name='u1')
        self.u2 = Value(generator.normal(loc=0.5, scale=0.5), name='u2')

        self.v = Value(generator.normal(loc=0.5, scale=0.5), name='v')

        self._parameters = [self.w0, self.w1, self.w2, self.u0, self.u1, self.u2, self.v]

    def parameters(self):
        return self._parameters
    
    def forward(self, x1, x2):
        h = activation(self.w0 + self.w1 * x1 + self.w2 * x2)
        y = activation(self.u0 + self.u1 * x1 + self.u2 * x2 + self.v * h)
        return y

def mse_loss(pred, target):
    return (pred - target)**2

def train_epoch(model, loss_fn, features, targets):
    for i in range(features.shape[0]):
        pred = model.forward(features[i,0].item(), features[i,1].item())
        loss = loss_fn(pred, targets[i].item()) / features.shape[0]
        loss.backward()

def to_logic(x: float) -> bool:
    return x > 0.5

def predict(model, features):
    result = []
    for i in range(features.shape[0]):
        pred = Value.unwrap(model.forward(features[i,0].item(), features[i,1].item()))
        result.append(pred)
    return result

def result_table(features, pred: list, targets: np.array):
    df = pd.DataFrame(features, columns=["in1", "in2"])
    df["pred_raw"] = pred
    df["pred"] = [to_logic(p) for p in pred]
    df["target"] = pd.Series(targets.flatten()).map(to_logic)
    return df

def calc_loss(loss_fn, pred: list, target: np.array):
    loss = 0.
    for i in range(len(pred)):
        loss += loss_fn(pred[i], target[i])
    return loss / len(pred)

def calc_accuracy(pred: list, target: np.array):
    correct = 0
    for i in range(len(pred)):
        if to_logic(pred[i]) == to_logic(target[i].item()):
            correct += 1
    return correct / len(pred)
