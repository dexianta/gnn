package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConsistentDims(t *testing.T) {
	assert.False(t, consistentShape([][]int{{2, 3}, {1, 3}}))
	assert.True(t, consistentShape([][]int{{2, 3}, {2, 3}}))
}

func TestParseDim(t *testing.T) {
	a := d2{{1, 2}, {2, 3}, {3, 4}}
	dim, err := parseShape(a, []int{})
	assert.Equal(t, dim, []int{3, 2})
	assert.Nil(t, err)

	a2 := d1{1, 2, 3}
	dim, err = parseShape(a2, []int{})
	assert.Equal(t, dim, []int{3})
	assert.Nil(t, err)

	a3 := d2{{1, 2}, {2}}
	_, err = parseShape(a3, []int{})
	assert.Equal(t, err, ErrInvalidShape)

	a4 := d3{
		{
			{1, 2, 3},
			{1, 2, 3},
		},
		{
			{1, 2, 3},
			{1, 2, 3},
		},
	}
	dim, err = parseShape(a4, []int{})
	assert.Equal(t, dim, []int{2, 2, 3})
	assert.Nil(t, err)

	a5 := d3{
		{
			{1, 2, 3},
			{1, 2, 3},
		},
		{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		},
	}
	_, err = parseShape(a5, []int{})
	assert.Equal(t, err, ErrInvalidShape)
}

func TestIndexConversion(t *testing.T) {
	shape := []int{3, 3, 3}
	for i := 0; i < 27; i++ {
		pos := toPos(i, shape)
		idx := toIndex(pos, shape)
		assert.Equal(t, idx, i)
	}
}

func TestMatmulShape(t *testing.T) {
	specs := []struct {
		a        Shape
		b        Shape
		expected Shape
		k        int
	}{
		{
			Shape{1, 2, 3},
			Shape{1, 3, 2},
			Shape{1, 2, 2},
			3,
		},
		{
			Shape{10, 3, 4},
			Shape{4, 1},
			Shape{10, 3, 1},
			4,
		},
		{
			Shape{10, 3, 4},
			Shape{10, 4, 5},
			Shape{10, 3, 5},
			4,
		},
	}

	for _, spec := range specs {
		actual, k := newShapeForMatMul(spec.a, spec.b)
		assert.Equal(t, spec.expected, actual)
		assert.Equal(t, k, spec.k)
	}
}

func TestValidMatmulShape(t *testing.T) {
	specs := []struct {
		a     Shape
		b     Shape
		valid bool
	}{
		{
			Shape{1, 2, 3},
			Shape{1, 3, 2},
			true,
		},
		{
			Shape{10, 3, 4},
			Shape{4, 1},
			true,
		},
		{
			Shape{10, 3, 4},
			Shape{10, 3, 5},
			false,
		},
		{
			Shape{10, 3, 4},
			Shape{1, 3, 5}, // should be able to broadcast in pytorch, but not supported now
			false,
		},
		{
			Shape{10, 3, 4},
			Shape{2, 4, 5}, // should be able to broadcast in pytorch, but not supported now
			false,
		},
	}

	for _, spec := range specs {
		e := validShapesForMatmul(spec.a, spec.b)
		if spec.valid {
			assert.Nil(t, e)
		} else {
			assert.NotNil(t, e)
		}
	}
}

func TestGetKpair(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := Shape{2, 3, 4, 5}
		b := Shape{3, 5, 3}

		c, k := newShapeForMatMul(a, b)
		assert.Equal(t, c, Shape{2, 3, 4, 3})
		assert.Equal(t, k, 5)

		iter := c.Iter()
		for iter.Next() {
			pos := iter.Step()
			pairs := getMatmulPairs(a, b, pos)
			assert.Len(t, pairs, 5)
		}
	})

	t.Run("", func(t *testing.T) {
		a := Shape{2, 3}
		b := Shape{3, 2}

		//  x,x,x     y,y
		//  x,x,x     y,y
		//            y,y

		c, k := newShapeForMatMul(a, b)
		assert.Equal(t, c, Shape{2, 2})
		assert.Equal(t, 3, k)

		iter := c.Iter()
		assert.Equal(t,
			getMatmulPairs(a, b, iter.Step()),
			[]matmulPair{
				{
					a: []int{0, 0},
					b: []int{0, 0},
				},
				{
					a: []int{0, 1},
					b: []int{1, 0},
				},
				{
					a: []int{0, 2},
					b: []int{2, 0},
				},
			})

		// 0,1
		assert.Equal(t,
			getMatmulPairs(a, b, iter.Step()),
			[]matmulPair{
				{
					a: []int{0, 0},
					b: []int{0, 1},
				},
				{
					a: []int{0, 1},
					b: []int{1, 1},
				},
				{
					a: []int{0, 2},
					b: []int{2, 1},
				},
			})
	})
}
