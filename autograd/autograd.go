package autograd

import (
	"fmt"
	"math"
)

type Op string

const (
	Add Op = "+"
	Sub Op = "-"
	Mul Op = "*"
	Div Op = "/"
	Pow Op = "^"
	Exp Op = "exp"
	Neg Op = "-"
)

type BinaryOp struct {
	op Op
	l  *Val // left
	r  *Val // right
}

type PowOp struct {
	v *Val
	p float64
}

type ExpOp struct {
	v *Val
}

type UnaryOp struct {
	op Op
	v  *Val
}

type NullOp struct{}

var nu = &NullOp{} // for leaf node

type Val struct {
	data float64
	grad float64

	prev any // BinaryOp or UnaryOp
}

func NewVal(data float64) *Val {
	return &Val{
		data: data,
		prev: nu,
	}
}

// Back Propagation is the process to find out the gradient of local variable with regard to the final output of interest:
// i.e. d<final result>/d<local variable>.
// for an example: ((a * b) + d) * e = f
// to find out the contribution of a to f (df/da), we can use the chain rule from calculus to trace back from the end
// df/da = df/df *                  | 1
//         df/d((a * b) + d) *      | e
//         d((a * b) + d)/d(a*b) *  | 1
//         d(a*b)/d(a)              | b
//
// final result as we can see, is just 1 * e * 1 * b = eb
//
// df/dd = df/df *                  | 1
//         df/d((a * b) + d) *      | e
//         d((a * b) + d)/dd        | 1
//
// again, final result as we can see, is just 1 * e * (a * b) = e
//
// it's easy to spot why we call it back propagation, since we always started from the root, which is always 1, and working
// our way backwards. And for the last step: we always arrive this expression:
//          <accumulated gradient from previous computation> * d<current expression>/d<variable of interest>
// following this same principal, you can derive all the df/d<whatever you want>
// this procedure seems tedious but has the advantage of being easy to implement as computer program, as opposite to
// what we are taught in calculus class (solve it by hand by expanding the expression)

// Backward will traverse the computation graph and populate the gradient of each node
// the special case are for the root, which is just 1
// externalGrad is needed to kick-off the traverse
func (v *Val) Backward(accumulatedGrad float64) {
	v.grad = accumulatedGrad
	switch o := v.prev.(type) {
	case *BinaryOp:
		switch o.op {
		case Add:
			// for addition, x + y
			// the way to calculate gradient is accumulated_grad * d(x + y)/dx = accumulated_grad
			o.l.grad += v.grad // if the Val were used multiple times, we need to accumulate the gradient
			o.r.grad += v.grad
			o.l.Backward(o.l.grad)
			o.r.Backward(o.r.grad)
		case Mul:
			// for multiplication, x * y
			// the way to calculate gradient is accumulated_grad * d(x*y)/dx = accumulated_grad * y
			o.l.grad += o.r.data * v.grad
			o.r.grad += o.l.data * v.grad
			o.l.Backward(o.l.grad)
			o.r.Backward(o.r.grad)
		}
	case *PowOp:
		// x^n --> nx^n-1
		o.v.grad += o.p * math.Pow(o.v.data, o.p-1) * v.grad
		o.v.Backward(o.v.grad)

	case *ExpOp:
		// e^x --> e^x
		o.v.grad += math.Exp(o.v.data) * v.grad
		o.v.Backward(o.v.grad)
	case *NullOp:
	default:
		panic(fmt.Errorf("invalid op for prev: %T", o))
	}
}

func (v *Val) Neg() *Val {
	return v.Mul(NewVal(-1.))
}

func (v *Val) Sub(o *Val) *Val {
	return v.Add(o.Neg())
}

func (v *Val) Add(o *Val) *Val {
	ret := &Val{}
	ret.data = v.data + o.data
	ret.prev = &BinaryOp{
		op: Add,
		l:  v,
		r:  o,
	}
	return ret
}

func (v *Val) Mul(o *Val) *Val {
	ret := &Val{}
	ret.data = v.data * o.data
	ret.prev = &BinaryOp{
		op: Mul,
		l:  v,
		r:  o,
	}
	return ret
}

func (v *Val) Div(o *Val) *Val {
	// a / b = a * (b ^ -1)
	return v.Mul(o.Pow(-1.))
}

// base with nature e
func (v *Val) Exp() *Val {
	ret := &Val{}
	ret.data = math.Exp(v.data)
	ret.prev = &ExpOp{
		v: v,
	}
	return ret
}

func (v *Val) Pow(p float64) *Val {
	ret := &Val{}
	ret.data = math.Pow(v.data, p)
	ret.prev = &PowOp{
		v: v,
		p: p,
	}
	return ret
}
