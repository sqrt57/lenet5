import math


class Value:
    def __init__(self, value, name=None, *, arguments=None, function=None):
        self.value = value
        self.name = name
        self.arguments: list[Value] | None = arguments
        self.function = function
        self.grad = 0.
        self.grad_back = 0.

    def wrap(x):
        if isinstance(x, Value):
            return x
        else:
            return Value(x)
    
    def unwrap(x):
        if isinstance(x, Value):
            return x.value
        else:
            return x
        
    def values_sorted_by_topological_order(self):
        visited = set()
        result = []
        def visit(value):
            if value in visited:
                return
            visited.add(value)
            result.append(value)
            if value.arguments is not None:
                for arg in value.arguments:
                    visit(arg)
        visit(self)
        return result
    
    def backward(self, grad=1):
        values = self.values_sorted_by_topological_order()

        for value in values:
            value.grad_back = 0.
        self.grad_back = grad

        for value in values:
            if value.function is not None:
                grads = value.function.derivative(*value.arguments)
                for arg, g in zip(value.arguments, grads):
                    arg.grad_back += g * value.grad_back

        for value in values:
            value.grad += value.grad_back

    def __repr__(self):
        if self.name is not None:
            return f"Value({self.value}, name='{self.name}')"
        else:
            return f"Value({self.value})"
        
    def __add__(self, other):
        return Value(self.value + Value.unwrap(other), arguments=[self, Value.wrap(other)], function=add)
    
    def __radd__(self, other):
        return Value(Value.unwrap(other) + self.value, arguments=[Value.wrap(other), self], function=add)
    
    def __sub__(self, other):
        return Value(self.value - Value.unwrap(other), arguments=[self, Value.wrap(other)], function=sub)
    
    def __rsub__(self, other):
        return Value(Value.unwrap(other) - self.value, arguments=[Value.wrap(other), self], function=sub)

    def __mul__(self, other):
        return Value(self.value * Value.unwrap(other), arguments=[self, Value.wrap(other)], function=mul)
    
    def __rmul__(self, other):
        return Value(Value.unwrap(other) * self.value, arguments=[Value.wrap(other), self], function=mul)
    
    def __truediv__(self, other):
        return Value(self.value / Value.unwrap(other), arguments=[self, Value.wrap(other)], function=div)
    
    def __rtruediv__(self, other):
        return Value(Value.unwrap(other) / self.value, arguments=[Value.wrap(other), self], function=div)
    
    def __neg__(self):
        return Value(-self.value, arguments=[self], function=neg)
    
    def __pow__(self, power):
        return Value(self.value ** Value.unwrap(power), arguments=[self, Value.wrap(power)], function=pow)
    
    def __rpow__(self, other):
        return Value(Value.unwrap(other) ** self.value, arguments=[Value.wrap(other), self], function=pow)

class Function1:
    def __init__(self, func, derivative, name=None):
        self._func = func
        self._derivative = derivative
        self.name = name
    
    def func(self, x):
        if isinstance(x, Value):
            return Value(self._func(x.value), arguments=[x], function=self)
        else:
            return self._func(x)
    
    def derivative(self, x):
        return [self._derivative(Value.unwrap(x))]

    def __call__(self, x):
        if isinstance(x, Value):
            return Value(self._func(x.value), arguments=[x], function=self)
        return self.func(x)

    def __repr__(self):
        if self.name is not None:
            return f"Function({self.func}, name='{self.name}')"
        else:
            return f"Function({self.func})"

class Function2:
    def __init__(self, func, derivative, name=None):
        self._func = func
        self._derivative = derivative
        self.name = name
    
    def func(self, x, y):
        if isinstance(x, Value) or isinstance(y, Value):
            x_val = Value.wrap(x)
            y_val = Value.wrap(y)
            return Value(self._func(x_val.value, y_val.value), arguments=[x_val, y_val], function=self)
        else:
            return self._func(x, y)
    
    def derivative(self, x, y):
        return self._derivative(Value.unwrap(x), Value.unwrap(y))

    def __call__(self, x, y):
        return self.func(x, y)

    def __repr__(self):
        if self.name is not None:
            return f"Function({self.func}, name='{self.name}')"
        else:
            return f"Function({self.func})"

class Add(Function2):
    def __init__(self):
        super().__init__(Add._func, Add._derivative, name='+')

    def _func(x, y):
        return x + y

    def _derivative(x, y):
        return 1., 1.

class Sub(Function2):    
    def __init__(self):
        super().__init__(Sub._func, Sub._derivative, name='-')

    def _func(x, y):
        return x - y

    def _derivative(x, y):
        return 1., -1.

class Mul(Function2):
    def __init__(self):
        super().__init__(Mul._func, Mul._derivative, name='*')

    def _func(x, y):
        return x * y

    def _derivative(x, y):
        return y, x

class Div(Function2):
    def __init__(self):
        super().__init__(Div._func, Div._derivative, name='/')

    def _func(x, y):
        return x / y

    def _derivative(x, y):
        return 1/y, -x/(y*y)
    
class Neg(Function1):
    def __init__(self):
        super().__init__(Neg._func, Neg._derivative, name='-')

    def _func(x):
        return -x

    def _derivative(x):
        return -1.

class Pow(Function2):
    def __init__(self):
        super().__init__(Pow._func, Pow._derivative, name='**')
    
    def _func(x, y):
        return x ** y
    
    def _derivative(x, y):
        return y * (x ** (y - 1)),  math.log(x) * (x ** y) if x > 0 else 0.

class Tanh(Function1):

    def __init__(self):
        super().__init__(Tanh._func, Tanh._derivative, name='tanh')

    def _func(x):
        return math.tanh(x)

    def _derivative(x):
        return 1 - math.tanh(x)**2

add = Add()
sub = Sub()
mul = Mul()
div = Div()
neg = Neg()
pow = Pow()
tanh = Tanh()

class SGD:
    def __init__(self, parameters: list[Value], *, lr=0.001):
        self.parameters = parameters.copy()
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.

    def step(self):
        for param in self.parameters:
            param.value -= self.lr * param.grad

class Adam:
    def __init__(self, parameters: list[Value], *, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters.copy()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0.] * len(self.parameters)
        self.v = [0.] * len(self.parameters)
        self.t = 0

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            g = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.value -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)

class AdamW:
    def __init__(self, parameters: list[Value], *, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.parameters = parameters.copy()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = [0.] * len(self.parameters)
        self.v = [0.] * len(self.parameters)
        self.t = 0

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            g = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.value -= self.lr * (m_hat / (math.sqrt(v_hat) + self.epsilon) + self.weight_decay * param.value)