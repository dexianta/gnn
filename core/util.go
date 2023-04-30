package core

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
