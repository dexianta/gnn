## Bad Neural Network

A simple implementation of basic neural network inspired by [micrograd](https://github.com/karpathy/micrograd).  
The APIs are influenced by pytorch

### Motivation
Neural network is a complex construct to approximate target output by 
modifying it's internal states through training to minimize difference 
between predicted output with actual known answer.

A very common technique is what's called back propagation to iteratively minimize this difference. 
Simply put, if we say an neural network is an function `f`, it's input `x`, predicted output `f(x) = y`, 
and the correct output is `Y`, then we need to find the right internal state of `f`, which minimize the loss :
`Y - y` or `Y - f(x)`.

Backward propagation was a very interesting idea which I only had a superficial understanding at the beginning. 
This repo aims to recreate a small neural network from scratch, to be able to solve the MNIST problem.

### Concept
I'm using pytorch as a style guide for the API, but as Python and Go has very different approach and capabilities.\
There's more effort to overcome for things we take for granted in Python, most notably: operator overloading and dynamic typing\

The fundamental abstraction is a `Val`, which stores a float(64) and the gradient with respect to the root.  
```go
type Val struct {
  data float64
  grad float64
  prev any // the operation that produce this val, can be BinaryOp | UnaryOp | PowOp | ExpOp
}
```

