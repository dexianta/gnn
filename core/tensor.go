package core

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
)

type S [2]int // for slice
type Shape []int
type Pos []int

func (s Shape) Equal(a Shape) bool {
	if len(s) != len(a) {
		return false
	}
	for i := range s {
		if s[i] != a[i] {
			return false
		}
	}
	return true
}

type ShapeIter struct {
	idx   int
	shape Shape
}

func (s *ShapeIter) Next() bool {
	return s.idx <= s.shape.MaxIdx()
}

// Step
// return current position, and increase
func (s *ShapeIter) Step() (ret Pos) {
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

func (s Shape) Valid(pos Pos) error {
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

func (t Tensor) GetBatchV(pos []Pos) (ret []*V) {
	for _, p := range pos {
		ret = append(ret, t.data[toIndex(p, t.Shape)])
	}
	return
}

func (t Tensor) GetV(pos Pos) *V {
	if len(pos) != len(t.Shape) {
		panic("invalid pos")
	}

	return t.data[toIndex(pos, t.Shape)]
}

func (t Tensor) Dim() int {
	return len(t.Shape)
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

func Zeros(dims ...int) Tensor {
	shape := Shape(dims)
	ret := Tensor{
		data:  make([]*V, shape.Cap()),
		Shape: shape,
	}
	for i := range ret.data {
		ret.data[i] = Vx(0)
	}
	return ret
}

func All(val float64, dims []int) Tensor {
	shape := Shape(dims)
	t := Tensor{
		data:  make([]*V, shape.Cap()),
		Shape: shape,
	}

	for i := range t.data {
		t.data[i] = Vx(val)
	}

	return t
}

func Ones(dims ...int) Tensor {
	shape := Shape(dims)
	t := Tensor{
		data:  make([]*V, shape.Cap()),
		Shape: shape,
	}

	for i := range t.data {
		t.data[i] = Vx(1)
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
		t.data[i] = Vx(rand.NormFloat64())
	}

	return t
}

func sortRank(a, b Tensor) (higher, lower Tensor) {
	if len(a.Shape) > len(b.Shape) {
		return a, b
	}
	return b, a
}

// only support a simpler version of broadcast
func canBroadcast(a, b Tensor) bool {
	highRank, lowRank := sortRank(a, b)
	if reflect.DeepEqual(lowRank.Shape, Shape{1}) {
		return true
	}

	diff := len(highRank.Shape) - len(lowRank.Shape)
	for i := len(lowRank.Shape) - 1; i >= 0; i-- {
		if lowRank.Shape[i] != highRank.Shape[i+diff] {
			return false
		}
	}
	return true
}

func basicOp(a, b *V, op string) *V {
	switch op {
	case "*":
		return a.Mul(b)
	case "+":
		return a.Add(b)
	case "/":
		return a.Div(b)
	case "-":
		return a.Sub(b)
	default:
		panic("invalid op")
	}
}

func broadcastOp(x, y Tensor, op string) (ret Tensor) {
	if x.Shape.Equal(y.Shape) {
		ret.Shape = x.Shape
		ret.data = make([]*V, len(x.data))
		for i := range ret.data {
			ret.data[i] = basicOp(x.data[i], y.data[i], op)
		}
		return
	}

	if canBroadcast(x, y) {
		h, _ := sortRank(x, y)
		ret.Shape = h.Shape // set the shape
		ret.data = make([]*V, len(h.data))
		for i := range ret.data {
			ret.data[i] = basicOp(x.data[i], y.data[i%len(y.data)], op)
		}
	} else {
		panic("cannot add")
	}

	return
}

func (t Tensor) Add(a Tensor) (ret Tensor) {
	return broadcastOp(t, a, "+")
}

func (t Tensor) Sub(a Tensor) (ret Tensor) {
	return broadcastOp(t, a, "-")
}

func (t Tensor) Mul(a Tensor) (ret Tensor) {
	return broadcastOp(t, a, "*")
}

func (t Tensor) Div(a Tensor) (ret Tensor) {
	return broadcastOp(t, a, "/")
}

// S means scalar
func (t Tensor) AddS(v float64) Tensor {
	for i := range t.data {
		t.data[i].Data = t.data[i].Data + v
	}
	return t
}

func (t Tensor) MulS(v float64) Tensor {
	for i := range t.data {
		t.data[i].Data = t.data[i].Data * v
	}
	return t
}

func (t Tensor) DivS(v float64) Tensor {
	return t.MulS(1. / v)
}

func (t Tensor) Grad() (ret []float64) {
	ret = make([]float64, len(t.data))
	for i := range t.data {
		ret[i] = t.data[i].Grad
	}
	return ret
}

func (t *Tensor) Equal(o Tensor) bool {
	if !reflect.DeepEqual(t.Shape, o.Shape) {
		return false
	}

	if len(t.data) != len(o.data) {
		return false
	}

	for i := range t.data {
		dff := math.Abs(t.data[i].Data - o.data[i].Data)
		if dff > 0.001 {
			fmt.Printf("inequality: %v, %v, dff: %v\n", t.data[i].Data, o.data[i].Data, dff)
			return false
		}
	}
	return true
}

func (t *Tensor) Loc(loc []int) float64 {
	return t.data[toIndex(loc, t.Shape)].Data
}

func (t *Tensor) At(pos []int) *V {
	Panic(t.Shape.Valid(pos))
	return t.data[toIndex(pos, t.Shape)]
}

func (t Tensor) String() string {
	return fmt.Sprintf("\n%s\n", buildString([]int{}, t.Shape, t.data, "data"))
}

func (t Tensor) PrintData() {
	fmt.Println(buildString([]int{}, t.Shape, t.data, "data"))
}

func (t Tensor) PrintGrad() {
	fmt.Println(buildString([]int{}, t.Shape, t.data, "grad"))
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

func toVerboseSlice(sl [][2]int, shape Shape) (ret [][2]int) {
	ret = make([][2]int, len(shape))
	for i := range shape {
		if i >= len(sl) {
			ret[i] = [2]int{0, shape[i]}
		} else {
			ret[i] = sl[i]
		}
	}
	return
}

func inRange(idx int, verboseRange [][2]int, shape Shape) (bool, []int) {
	if len(verboseRange) != len(shape) {
		panic("verbose slice range is not valid")
	}

	pos := toPos(idx, shape)
	newPos := make([]int, len(pos))
	for i := range pos {
		if pos[i] >= verboseRange[i][1] || pos[i] < verboseRange[i][0] {
			return false, []int{}
		}
		newPos[i] = pos[i] - verboseRange[i][0]
	}

	return true, newPos
}

func (t Tensor) Slice(sl ...[2]int) (ret Tensor) {
	if len(sl) > len(t.Shape) {
		panic("invalid slicing")
	}

	for i := range sl {
		if sl[i][0] < 0 || sl[i][1] < 0 {
			panic("negative slice not supported")
		}
		if sl[i][0] > sl[i][1] {
			panic("invalid slicing: start>end")
		}
		if sl[i][1] > t.Shape[i] {
			panic(fmt.Sprintf("out of range index: %v, shape: %v", sl, t.Shape))
		}
	}

	// initialize new Tensor
	// new shape
	ret.Shape = make(Shape, len(t.Shape))
	for i := range t.Shape {
		if i >= len(sl) {
			ret.Shape[i] = t.Shape[i]
		} else {
			ret.Shape[i] = sl[i][1] - sl[i][0]
		}
	}
	ret.data = make([]*V, ret.Shape.Cap())

	verboseSlice := toVerboseSlice(sl, t.Shape)
	for i := range t.data {
		if ok, pos := inRange(i, verboseSlice, t.Shape); ok {
			ret.data[toIndex(pos, ret.Shape)] = t.data[i]
		}
	}

	return
}

func buildString(pos, shape []int, data []*V, field string) string {
	var tmp []string
	if len(pos) == len(shape)-1 {
		pos = append(pos, 0)
		start := toIndex(pos, shape)
		for i := 0; i < shape[len(shape)-1]; i++ {
			switch field {
			case "data":
				tmp = append(tmp, fmt.Sprintf("%.4f", data[start+i].Data))
			case "grad":
				tmp = append(tmp, fmt.Sprintf("%.4f", data[start+i].Grad))
			default:
				panic("invalid field to build string")
			}
		}

		return fmt.Sprint(tmp)
	}

	// len(pos) is the depth of recursion
	for i := 0; i < shape[len(pos)]; i++ {
		tmp = append(tmp, buildString(append(pos, i), shape, data, field))
	}
	return fmt.Sprintf("[%s]", strings.Join(tmp, strings.Repeat("\n", len(shape)-len(pos)-1)))
}
