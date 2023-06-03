package core

import (
	"errors"
	"fmt"
	"math"
)

func nrange(n int) (ret []int) {
	for i := 0; i < n; i++ {
		ret = append(ret, i)
	}
	return
}

func mul[T ~int | ~float64](arr []T) (ret T) {
	ret = 1
	for _, t := range arr {
		ret *= t
	}
	return
}

func sum[T ~int | ~float64](arr []T) (ret T) {
	for _, t := range arr {
		ret += t
	}
	return
}

func Panic(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func Range(start, end, step float64) (ret []float64) {
	for s := start; s < end; s += step {
		ret = append(ret, s)
	}
	return
}

func Map[T any](arr []float64, f func(x float64) T) (ret []T) {
	ret = make([]T, len(arr))
	for i := range arr {
		ret[i] = f(arr[i])
	}
	return ret
}

func DimIter(pos Pos, dim int, shape Shape) (ret []Pos) {
	if dim > len(pos) {
		panic(fmt.Sprintf("invalid dim: %d for pos: %v", dim, pos))
	}
	if err := shape.Valid(pos); err != nil {
		panic(err.Error())
	}

	for i := 0; i < shape[dim]; i++ {
		var tmp = make(Pos, len(pos))
		copy(tmp, pos)
		tmp[dim] = i
		ret = append(ret, tmp)
	}
	return
}

func EqualFloatArray(a, b []float64, err float64) error {
	err = math.Abs(err)
	if len(a) != len(b) {
		return errors.New("different length")
	}

	for i := range a {

		if math.Abs(a[i]-b[i]) > err {
			return fmt.Errorf("%f - %f > %f", a[i], b[i], err)
		}
	}
	return nil
}
