package common

import (
	"errors"
	"fmt"
)

type d1 []float64
type d2 [][]float64
type d3 [][][]float64
type d4 [][][][]float64
type dimension interface {
	~[]float64 | ~[][]float64 | ~[][][]float64 | ~[][][][]float64
}

func consistentDims(dims [][]int) bool {
	var key string
	for _, d := range dims {
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
	ErrInvalidDimension = errors.New("invalid dimension")
)

// parseDim parses the dimension of a given tensor
func parseDim[T dimension](arr T, dim []int) ([]int, error) {
	switch v := any(arr).(type) {
	case d1:
		return append(dim, len(v)), nil
	case d2:
		if len(v) == 0 {
			return dim, ErrInvalidDimension
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseDim(d1(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentDims(dims) {
			return dim, ErrInvalidDimension
		}
		return append([]int{len(v)}, dims[0]...), nil
	case d3:
		if len(v) == 0 {
			return dim, ErrInvalidDimension
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseDim(d2(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentDims(dims) {
			return dim, ErrInvalidDimension
		}
		return append([]int{len(v)}, dims[0]...), nil
	case d4:
		if len(v) == 0 {
			return dim, ErrInvalidDimension
		}
		var dims [][]int
		for _, vv := range v {
			d, err := parseDim(d3(vv), dim)
			if err != nil {
				return dim, err
			}
			dims = append(dims, d)
		}
		if !consistentDims(dims) {
			return dim, ErrInvalidDimension
		}
		return append([]int{len(v)}, dims[0]...), nil
	default:
		panic("invalid type")
	}
}
