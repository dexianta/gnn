package core

func Softmax(a Tensor, dim int) (ret Tensor) {
	if dim >= a.Dim() {
		panic("invalid dim >= a.Dim()")
	}

	ret.Shape = a.Shape
	ret.data = make([]*V, len(a.data))

	for i := range ret.data {
		pos := toPos(i, a.Shape)
		poses := DimIter(pos, dim, a.Shape)

		var tmp []*V
		for _, p := range poses {
			tmp = append(tmp, a.data[toIndex(p, a.Shape)])
		}

		fn := func(v *V) *V {
			return v.Exp()
		}
		num := a.data[i].Exp()
		denum := Sum(MapV(tmp, fn))
		ret.data[i] = num.Div(denum)
	}

	return
}

func LogSoftmax(a Tensor, dim int) (ret Tensor) {
	ret = Softmax(a, dim)
	for i := range ret.data {
		ret.data[i] = ret.data[i].Log()
	}
	return ret
}
