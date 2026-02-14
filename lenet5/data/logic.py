import numpy as np

features = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
targets_and = np.array([[0.],[0.],[0.],[1.]])
targets_or = np.array([[0.],[1.],[1.],[1.]])
targets_xor = np.array([[0.],[1.],[1.],[0.]])
targets_and_not = np.array([[1.],[1.],[1.],[0.]])
targets_or_not = np.array([[1.],[0.],[0.],[0.]])
