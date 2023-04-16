package common

import (
	"errors"
	"fmt"
)

type d1 []float64
type d2 [][]float64
type d3 [][][]float64
type d4 [][][][]float64

// ndb -> n-d array bound
type ndb interface {
	~[]float64 | ~[][]float64 | ~[][][]float64 | ~[][][][]float64
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

func toIndex(pos, shape []int) (ret int) {
	if len(pos) != len(shape) {
		panic("bad shape")
	}
	for i := range pos {
		stride := 1
		for j := i + 1; j < len(shape); j++ {
			stride *= shape[j]
		}
		ret += pos[i] * stride
	}
	return
}

func buildNdArrayIntoSingleDim[T ndb](arr T, shape []int, data []*Var) {
	// dim should match with the shape of arr
	switch v := any(arr).(type) {
	case d1:
		for _, i := range nrange(shape[0]) {
			data[i] = NewVar(v[i])
		}
	case d2:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				data[toIndex([]int{i, j}, shape)] = NewVar(v[i][j])
			}
		}
	case d3:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				for _, k := range nrange(shape[2]) {
					data[toIndex([]int{i, j, k}, shape)] = NewVar(v[i][j][k])
				}
			}
		}
	case d4:
		for _, i := range nrange(shape[0]) {
			for _, j := range nrange(shape[1]) {
				for _, k := range nrange(shape[2]) {
					for _, l := range nrange(shape[3]) {
						data[toIndex([]int{i, j, k, l}, shape)] = NewVar(v[i][j][k][l])
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
