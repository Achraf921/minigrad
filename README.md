# minigrad

A minimal autograd engine and neural net library I built from the ground up, following
[Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1).

The core idea is a `Value` object that wraps a number and remembers how it was computed.
Every atomic operation (`+`, `-`, `*`, `/`, `**k`, `exp`, `tanh`) knows its own local
derivative, and calling `backward()` on the output applies the chain rule through the
whole computation graph (reverse post-order DFS on the DAG (i.e topological sort)) to fill in every gradient.

On top of that sits a small neural net library (`Neuron`, `Layer`, `MLP`) with a
PyTorch-like API, and a `train` method that does the full loop: forward pass, loss,
zero the grads, backward pass, nudge the weights.

## Usage

```python
from minigrad import MLP

data = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
model.train(iter=50, step=0.05, data=data, expec=targets)
```

There is also a small graphviz visualizer in `minigrad.viz` to draw the computation
graph of any `Value` (requires graphviz installed).

## Notes and limitations

- Loss is sum of squared errors, and training is plain gradient descent
- Only supports array shaped data, no tensors or batching
- Learning rate and number of iterations are left for you to pick, choose wisely
- I left the jupyter notebook notes in the scratchwork folder that shows the learning process
- Everything is scalar-valued under the hood, this is just for learning
