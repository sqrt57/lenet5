import abc

class SimpleModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, features, labels):
        pass
    
    @abc.abstractmethod
    def forward(self, features):
        pass
