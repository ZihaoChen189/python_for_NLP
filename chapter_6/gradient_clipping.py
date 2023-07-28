import numpy as np

# simple example
dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]  # store them in one list

max_norm = 5.0  # limit

def clip_grads(grads, max_norm):
    total_norm = 0  # ready
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)  # the L2-norm number

    # check whether the final L2-norm number larger than the limit "max_norm"
    rate = max_norm / (total_norm + 1e-6) 
    if rate < 1:
        for grad in grads:
            grad *= rate  # if really larger, must be updated
            