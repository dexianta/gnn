package autograd

import (
	"fmt"
	"math"
)

type Op string

const (
	Add  Op = "+"
	Sub  Op = "-"
	Mul  Op = "*"
	Div  Op = "/"
	Pow  Op = "^"
	Exp  Op = "exp"
	ReLu Op = "relu"
)

type BinaryOp struct {
	op Op
	l  *Var // left
	r  *Var // right
}

type PowOp struct {
	v *Var
	p float64
}

type ExpOp struct {
	v *Var
}

type UnaryOp struct {
	op Op
	v  *Var
}

type NullOp struct{}

var nu = &NullOp{} // for leaf node

// Var is a variable, in a proper implementation, it should be a tensor
// it can be just a simple number (leaf node) without prev:
//
//	Var{data: 3, prev: nu}
//
// or, it's produced by some previous operation by prev:
//
//	Var{data: 3, prev: BinaryOp{op: Add, l: v1, r: v2}}
type Var struct {
	data float64
	grad float64

	prev any // BinaryOp / UnaryOp / PowOp / ExpOp / NullOp (leaf node)
}

func NewVar(data float64) *Var {
	return &Var{
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
func (v *Var) Backward(accumulatedGrad float64) {
	v.grad = accumulatedGrad
	switch pr := v.prev.(type) {
	case *BinaryOp:
		switch pr.op {
		case Add:
			// for addition, x + y
			// the way to calculate gradient is accumulated_grad * d(x + y)/dx = accumulated_grad
			pr.l.grad += v.grad // if the Val were used multiple times, we need to accumulate the gradient
			pr.r.grad += v.grad
			pr.l.Backward(pr.l.grad)
			pr.r.Backward(pr.r.grad)
		case Mul:
			// for multiplication, x * y
			// the way to calculate gradient is accumulated_grad * d(x*y)/dx = accumulated_grad * y
			pr.l.grad += pr.r.data * v.grad
			pr.r.grad += pr.l.data * v.grad
			pr.l.Backward(pr.l.grad)
			pr.r.Backward(pr.r.grad)
		}
	case *PowOp:
		// x^n --> nx^n-1
		pr.v.grad += pr.p * math.Pow(pr.v.data, pr.p-1) * v.grad
		pr.v.Backward(pr.v.grad)

	case *ExpOp:
		// e^x --> e^x
		pr.v.grad += math.Exp(pr.v.data) * v.grad
		pr.v.Backward(pr.v.grad)
	case *UnaryOp:
		switch pr.op {
		case ReLu:
			if v.data > 0 {
				pr.v.grad += v.grad
			} else {
				pr.v.grad += 0 // for readability
			}
			pr.v.Backward(pr.v.grad)
		default:
			panic(fmt.Errorf("invalid op for UnaryOp: %T", pr.op))
		}
	case *NullOp:
	default:
		panic(fmt.Errorf("invalid op for prev: %T", pr))
	}
}

func (v *Var) Neg() *Var {
	return v.Mul(NewVar(-1.))
}

func (v *Var) Sub(o *Var) *Var {
	return v.Add(o.Neg())
}

func (v *Var) Add(o *Var) *Var {
	ret := &Var{}
	ret.data = v.data + o.data
	ret.prev = &BinaryOp{
		op: Add,
		l:  v,
		r:  o,
	}
	return ret
}

func (v *Var) Mul(o *Var) *Var {
	ret := &Var{}
	ret.data = v.data * o.data
	ret.prev = &BinaryOp{
		op: Mul,
		l:  v,
		r:  o,
	}
	return ret
}

func (v *Var) Div(o *Var) *Var {
	// a / b = a * (b ^ -1)
	return v.Mul(o.Pow(-1.))
}

// base with nature e
func (v *Var) Exp() *Var {
	ret := &Var{}
	ret.data = math.Exp(v.data)
	ret.prev = &ExpOp{
		v: v,
	}
	return ret
}

func (v *Var) Pow(p float64) *Var {
	ret := &Var{}
	ret.data = math.Pow(v.data, p)
	ret.prev = &PowOp{
		v: v,
		p: p,
	}
	return ret
}

func (v *Var) ReLu() *Var {
	ret := &Var{
		prev: &UnaryOp{
			op: ReLu,
			v:  v,
		},
	}
	if v.data < 0 {
		ret.data = 0
	} else {
		ret.data = v.data
	}
	return ret
}
