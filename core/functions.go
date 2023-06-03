package core

// Softmax only applies to the -1 dimension
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
		ret.data[i] = a.data[i].Exp().Div(Sum(MapV(tmp, fn)))
	}

	return
}
