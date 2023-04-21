package common

type Shape []int

type ShapeIter struct {
	cur    []int
	shape  []int
	weight []int
}

func newShapeIter(cur, shape []int) (ret ShapeIter) {
	if len(cur) != len(shape) {
		panic("invalid dimension")
	}

	ret.weight = make([]int, len(cur))
	for i := range shape {
		ret.weight[i] = Product(shape[i:])
	}

	ret.cur = cur
	ret.shape = shape
	return
}

func (s Shape) IterFrom(start []int) ShapeIter {
	return newShapeIter(start, s)
}

func (s Shape) Iter() ShapeIter {
	return newShapeIter(make([]int, len(s)), s)
}
