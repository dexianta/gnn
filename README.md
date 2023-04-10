## Tiny Go Neural Network

A simple implementation of basic neural network inspired by [micrograd](https://github.com/karpathy/micrograd).  
The APIs are influenced by pytorch

Side note:  
Back propagation is a fundamental technique underlays most of the deep learning framework. For most of cases, you don't need to understand it to train a neural net.  
But It's never a bad thing to have a solid understanding on the subject, what's better than writing it from scratch?

### Concept
the current implementation is a special case of tensor with just one value, shape of (1,1)
The fundamental abstraction is a `Val`, which stores a float(64) and the gradient with respect to the root.  
```go
type Val struct {
  data float64
  grad float64
  prev any // the operation that produce this val, can be BinaryOp | UnaryOp | PowOp | ExpOp
}
```

