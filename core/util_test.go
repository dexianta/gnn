package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDimIter(t *testing.T) {
	pos := DimIter(Pos{0, 1, 1}, 0, Shape{4, 3, 2})
	assert.Equal(t, pos,
		[]Pos{{0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {3, 1, 1}})

	pos = DimIter(Pos{1, 2, 1}, 0, Shape{4, 3, 2})
	assert.Equal(t, pos,
		[]Pos{{0, 2, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1}})

	pos = DimIter(Pos{1, 2, 1}, 0, Shape{4, 3, 2})
	assert.Equal(t, pos,
		[]Pos{{0, 2, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1}})
}
