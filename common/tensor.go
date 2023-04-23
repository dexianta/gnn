package common

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
)

type Shape []int

type ShapeIter struct {
	idx   int
	shape Shape
}

func (s *ShapeIter) Next() bool {
	return s.idx <= s.shape.MaxIdx()
}

// Step
// return current position, and increase
func (s *ShapeIter) Step() (ret []int) {
	ret = toPos(s.idx, s.shape)
	s.idx++
	return
}

func (s Shape) Cap() int {
	return mul(s)
}

func (s Shape) MaxIdx() int {
	return mul(s) - 1 // 0 base
}

func (s Shape) Valid(pos []int) error {
	e := fmt.Errorf("invalid pos %v, shape: %v", pos, s)
	if len(pos) != len(s) {
		return e
	}

	for i := range pos {
		if pos[i] >= s[i] {
			return e
		}
	}

	return nil
}

func (s Shape) IterFrom(cur []int) ShapeIter {
	if len(cur) != len(s) {
		panic("invalid shape/cur")
	}

	return ShapeIter{
		idx:   toIndex(cur, s),
		shape: s,
	}
}

func (s Shape) Iter() ShapeIter {
	return ShapeIter{
		idx:   0,
		shape: s,
	}
}

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
	var data = make([]*V, mul(shape))
	buildNdArrayIntoSingleDim(arr, shape, data)
	ret.data = data
	return
}

func Ones(dims ...int) Tensor {
	shape := Shape(dims)
	t := Tensor{
		data:  make([]*V, shape.Cap()),
		Shape: shape,
	}

	for i := range t.data {
		t.data[i] = &V{
			data: 1,
		}
	}

	return t
}

func Randn(dims ...int) Tensor {
	shape := Shape(dims)
	t := Tensor{
		data:  make([]*V, shape.Cap()),
		Shape: shape,
	}

	for i := range t.data {
		t.data[i] = &V{
			data: rand.NormFloat64(),
		}
	}

	return t
}

func (t Tensor) Add(v float64) (ret Tensor) {
	ret.Shape = t.Shape
	ret.data = t.data

	for i := range ret.data {
		ret.data[i].data += v
	}
	return
}

func (t Tensor) Mul(v float64) (ret Tensor) {
	ret.Shape = t.Shape
	ret.data = t.data

	for i := range ret.data {
		ret.data[i].data *= v
	}
	return
}

func (t *Tensor) Equal(o Tensor) bool {
	if !reflect.DeepEqual(t.Shape, o.Shape) {
		return false
	}

	if len(t.data) != len(o.data) {
		return false
	}

	for i := range t.data {
		if t.data[i].data != o.data[i].data {
			return false
		}
	}
	return true
}

func (t *Tensor) Loc(loc []int) float64 {
	return t.data[toIndex(loc, t.Shape)].data
}

func (t *Tensor) At(pos []int) *V {
	Panic(t.Shape.Valid(pos))
	return t.data[toIndex(pos, t.Shape)]
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

	// ===========================
	// normal matrix multiplication
	// ===========================
	newShape, _ := newShapeForMatMul(t.Shape, o.Shape)
	ret.Shape = newShape
	ret.data = make([]*V, newShape.Cap()) // initialize

	iter := newShape.Iter()
	for iter.Next() {
		pos := iter.Step()
		idx := toIndex(pos, newShape)
		ps := getMatmulPairs(t.Shape, o.Shape, pos)
		var v []*V
		for _, p := range ps {
			v = append(v, t.At(p.a).Mul(o.At(p.b)))
		}
		ret.data[idx] = Sum(v)
	}
	return
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
