package common

func Sum[T ~int | ~float64](arr []T) (ret T) {
	for _, t := range arr {
		ret += t
	}
	return
}
