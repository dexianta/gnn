## Tiny Go Neural Network

A simple implementation of basic neural network inspired by [micrograd](https://github.com/karpathy/micrograd).

### Concept
The fundamental abstraction is a `Val`, which stores a float(64) and the gradient with respect to the root.  
This is a special case to the generalized version of tensor (has just one value).

