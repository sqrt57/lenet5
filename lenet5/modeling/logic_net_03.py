import numpy as np
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
    def __init__(self, generator: np.random.Generator = np.random.default_rng()):
        def rand():
            return generator.normal(loc=0.5, scale=0.1)
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