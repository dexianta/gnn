package core

import (
	"errors"
	"fmt"
)

type d1 []float64
type d2 [][]float64
type d3 [][][]float64
type d4 [][][][]float64
type ud1 []uint8
type ud2 [][]uint8

func errInvalidShape(a, b Shape) error {
	return fmt.Errorf("invalid shape: a(%v), b(%v)", a, b)
}

// ndb -> n-d array bound
type ndb interface {
	~[]float64 | ~[][]float64 | ~[][][]float64 | ~[][][][]float64 | ~[]uint8 | ~[][]uint8
}

func consistentShape(shape [][]int) bool {
	var key string
	for _, d := range shape {
		if key == "" {
			key = fmt.Sprint(d)
			continue
		}
		if key != fmt.Sprint(d) {
			return false
		}
	}
	return true
}

var (
	ErrInvalidShape = errors.New("invalid shape")
)

// (2,2,2) shape
// (0, 0, 0) -> 0
// (0, 0, 1) -> 1
// (0, 1, 0) -> 2
// (0, 1, 1) -> 3
// (1, 0, 0) -> 4
// (1, 0, 1) -> 5
func toPos(idx int, shape []int) []int {
	result := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		result[i] = idx % shape[i]
		idx = idx / shape[i]
	}
	return result
}

func toIndex(pos, shape []int) (ret int) {
	if len(pos) != len(shape) {
		panic("bad shape")
	}
	// starting from rightmost (LSB)

	for i := len(pos) - 1; i >= 0; i-- {
		if i == len(pos)-1 {
			ret = pos[i]
		} else {
			ret += pos[i] * mul(shape[i+1:])
		}
	}
	return
}

func buildNdArrayIntoSingleDim[T ndb](arr T, shape []int, data []*V) {
	// dim should match with the shape of arr
	switch v := any(arr).(type) {
	case ud1:
		for _, i := range nrange(shape[0]) {
			data[i] = Vx(float64(v[i]))
		}
	case ud2:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				data[toIndex([]int{i, j}, shape)] = Vx(float64(v[i][j]))
			}
		}
	case d1:
		for _, i := range nrange(shape[0]) {
			data[i] = Vx(v[i])
		}
	case d2:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				data[toIndex([]int{i, j}, shape)] = Vx(v[i][j])
			}
		}
	case d3:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				for _, k := range nrange(shape[2]) {
					data[toIndex([]int{i, j, k}, shape)] = Vx(v[i][j][k])
				}
			}
		}
	case d4:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				for _, k := range nrange(shape[2]) {
					for _, l := range nrange(shape[3]) {
						data[toIndex([]int{i, j, k, l}, shape)] = Vx(v[i][j][k][l])
					}
				}
			}
		}
	default:
		panic(ErrInvalidShape)
	}
}

// parseShape parses the dimension of a given tensor
func parseShape[T ndb](arr T, dim []int) ([]int, error) {
	switch v := any(arr).(type) {
	case d1:
		//==============
		return append(dim, len(v)), nil

	case d2:
		//==============
		if len(v) == 0 {
			return dim, ErrInvalidShape
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseShape(d1(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentShape(dims) {
			return dim, ErrInvalidShape
		}
		return append([]int{len(v)}, dims[0]...), nil

	case d3:
		//==============
		if len(v) == 0 {
			return dim, ErrInvalidShape
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseShape(d2(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentShape(dims) {
			return dim, ErrInvalidShape
		}
		return append([]int{len(v)}, dims[0]...), nil

	case d4:
		//==============
		if len(v) == 0 {
			return dim, ErrInvalidShape
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseShape(d3(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentShape(dims) {
			return dim, ErrInvalidShape
		}
		return append([]int{len(v)}, dims[0]...), nil

	default:
		panic("invalid type")
	}
}

func validShapesForMatmul(a, b Shape) error {
	if len(a) < 2 || len(b) < 2 {
		return errInvalidShape(a, b)
	}
	// check the basic matmul dimension
	if a[len(a)-1] != b[len(b)-2] {
		return errInvalidShape(a, b)
	}

	// check the batch multiply dimension
	for i := len(b) - 3; i >= 0; i-- {
		if b[i] != a[i+len(a)-len(b)] {
			return errInvalidShape(a, b)
		}
	}
	return nil
}

// only support a reduced version of the matrix multiplication
func newShapeForMatMul(a, b Shape) (ret Shape, k int) {
	// the most common matrix usecase (for me)
	// not taking care of broadcasting, I don't find it nature anyway
	if len(a) < len(b) {
		a, b = b, a
	}

	// other types of configuration
	// e.g. [2,3] @ [3] (lower dimension stuff)

	// batch or normal matmul
	Panic(validShapesForMatmul(a, b))
	ret = make([]int, len(a))
	copy(ret, a)

	ret[len(a)-1] = b[len(b)-1]
	k = a[len(a)-1]
	return
}

type matmulPair struct {
	a []int
	b []int
}

// sum(a(i,k) * b(k,j))
func getMatmulPairs(a, b Shape, cPos []int) (ret []matmulPair) {

	if len(a) < len(b) {
		a, b = b, a
	}

	Panic(validShapesForMatmul(a, b))
	// assumption:
	// - len(a) >= 2 && len(b) >= 2
	// - len(a) == len(cPos)
	// - len(b) <= len(cPos)
	// - a[len(a)-2] == b[len(b)-1]
	k := a[len(a)-1]
	for i := 0; i < k; i++ {
		aa := append([]int{}, cPos...)
		aa[len(a)-1] = i
		bb := append([]int{}, cPos[len(cPos)-len(b):]...)
		bb[len(b)-2] = i
		ret = append(ret, matmulPair{
			a: aa,
			b: bb,
		})
	}

	return
}
