package common

// Tensor is a n-dimensional array
// implemented with a single dimensional array with indexing tricks
type Tensor struct {
	data []*Var
	dim  []int
}

func newTensor(arr any) (ret Tensor) {
	return
}

func NewTensorRand(dim ...int) (ret Tensor) {
	return
}
