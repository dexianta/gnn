package common

import (
	"fmt"
	"strings"
)

// Tensor is a n-dimensional array
// implemented with a single dimensional array with indexing tricks
type Tensor struct {
	data  []*V
	Shape Shape
}

func NewTensor[T ndb](arr T) (ret Tensor) {
	shape, err := parseShape(arr, []int{})
	if err != nil {
		panic(err)
	}

	ret.Shape = shape
	var data = make([]*V, Product(shape))
	buildNdArrayIntoSingleDim(arr, shape, data)
	ret.data = data
	return
}

func (t *Tensor) Loc(loc []int) float64 {
	return t.data[toIndex(loc, t.Shape)].data
}

func (t Tensor) String() string {
	return fmt.Sprintf("\n%s\n", buildString([]int{}, t.Shape, t.data))
}

// matrix multiplication
func (t Tensor) Matmul(o Tensor) (ret Tensor) {
	// dot product
	if len(t.Shape) == 1 && len(o.Shape) == 1 && t.Shape[0] == o.Shape[0] {
		var data = make([]*V, t.Shape[0])
		for i := range t.data {
			data[i] = t.data[i].Mul(o.data[i])
		}
		ret.data = data
		ret.Shape = t.Shape
		return
	}

	// normal matrix multiplication

	panic("invalid operation")
}

func buildString(pos, shape []int, data []*V) string {
	var tmp []string
	if len(pos) == len(shape)-1 {
		pos = append(pos, 0)
		start := toIndex(pos, shape)
		for i := 0; i < shape[len(shape)-1]; i++ {
			tmp = append(tmp, fmt.Sprint(data[start+i]))
		}

		return fmt.Sprint(tmp)
	}

	// len(pos) is the depth of recursion
	for i := 0; i < shape[len(pos)]; i++ {
		tmp = append(tmp, buildString(append(pos, i), shape, data))
	}
	return fmt.Sprintf("[%s]", strings.Join(tmp, strings.Repeat("\n", len(shape)-len(pos)-1)))
}
