package common

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToIndex(t *testing.T) {
	tn := NewTensor(d2{{1, 2, 3}, {4, 5, 6}})
	fmt.Println(tn)
}

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
		a        []int
		b        []int
		expected []int
	}{
		{
			[]int{1, 2, 3},
			[]int{1, 3, 2},
			[]int{1, 2, 2},
		},
		{
			[]int{10, 3, 4},
			[]int{4, 1},
			[]int{10, 3, 1},
		},
		{
			[]int{10, 3, 4},
			[]int{10, 4, 5},
			[]int{10, 3, 5},
		},
	}

	for _, spec := range specs {
		assert.Equal(t, spec.expected, newShapeForMatMul(spec.a, spec.b))
	}
}
