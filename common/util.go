package common

func nrange(n int) (ret []int) {
	for i := 0; i < n; i++ {
		ret = append(ret, i)
	}
	return
}

func Product[T ~int | ~float64](arr []T) (ret T) {
	ret = 1
	for _, t := range arr {
		ret *= t
	}
	return
}

func Sum[T ~int | ~float64](arr []T) (ret T) {
	for _, t := range arr {
		ret += t
	}
	return
}
