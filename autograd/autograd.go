package autograd

type Op string

const (
	Add Op = "+"
	Sub Op = "-"
	Mul Op = "*"
	Div Op = "/"
	Pow Op = "^"
)

type BinaryOp struct {
	op Op
	l  *Val // left
	r  *Val // right
}

type UnaryOp struct {
	op Op
	v  *Val
}

type Val struct {
	data float64
	grad float64

	prev any // BinaryOp or UnaryOp
}

func NewVal(data float64) *Val {
	return &Val{
		data: data,
	}
}

// Backward will traverse the computation graph and populate the gradient of each node
// the special case are for the root, which is just 1
// externalGrad is needed to kick-off the traverse
func (v *Val) Backward(externalGrad float64) {
	v.grad = externalGrad
	switch o := v.prev.(type) {
	case BinaryOp:
		switch o.op {
		case Add:
			o.l.grad += v.grad
			o.r.grad += v.grad
			o.l.Backward(o.l.grad)
			o.r.Backward(o.r.grad)
			v.prev = o
		case Mul:
		case Pow:
		}
	case UnaryOp:
	}
}

func (v *Val) Sub(o *Val) Val {
	no := o
	no.data = no.data * (-1)
	return v.Add(no)
}

func (v *Val) Add(o *Val) (ret Val) {
	ret.data = v.data + o.data
	ret.prev = BinaryOp{
		op: Add,
		l:  v,
		r:  o,
	}
	return
}

func (v *Val) Mul(o *Val) (ret Val) {
	ret.data = v.data * o.data
	ret.prev = BinaryOp{
		op: Mul,
		l:  v,
		r:  o,
	}
	return
}
