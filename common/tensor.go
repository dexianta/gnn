package common

import (
	"fmt"
	"strings"
)

// Tensor is a n-dimensional array
// implemented with a single dimensional array with indexing tricks
type Tensor struct {
	data  []*Var
	shape []int
}

func NewTensor[T ndb](arr T) (ret Tensor) {
	shape, err := parseShape(arr, []int{})
	if err != nil {
		panic(err)
	}

	ret.shape = shape
	var data = make([]*Var, Product(shape))
	buildNdArrayIntoSingleDim(arr, shape, data)
	ret.data = data
	return
}

func (t *Tensor) Loc(idx ...int) float64 {
	return t.data[toIndex(idx, t.shape)].data
}

func (t Tensor) String() string {
	return fmt.Sprintf("\n%s\n", buildString([]int{}, t.shape, t.data))
}

func buildString(pos, shape []int, data []*Var) string {
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
