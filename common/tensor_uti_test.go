package common

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConsistentDims(t *testing.T) {
	assert.False(t, consistentDims([][]int{{2, 3}, {1, 3}}))
	assert.True(t, consistentDims([][]int{{2, 3}, {2, 3}}))
}

func TestParseDim(t *testing.T) {
	a := d2{{1, 2}, {2, 3}, {3, 4}}
	dim, err := parseDim(a, []int{})
	assert.Equal(t, dim, []int{3, 2})
	assert.Nil(t, err)

	a2 := d1{1, 2, 3}
	dim, err = parseDim(a2, []int{})
	assert.Equal(t, dim, []int{3})
	assert.Nil(t, err)

	a3 := d2{{1, 2}, {2}}
	_, err = parseDim(a3, []int{})
	assert.Equal(t, err, ErrInvalidDimension)

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
	dim, err = parseDim(a4, []int{})
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
	_, err = parseDim(a5, []int{})
	assert.Equal(t, err, ErrInvalidDimension)
}
