package common

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTensor(t *testing.T) {
	tn := NewTensor(d3{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}})

	s := tn.Shape
	v := 1.0
	assert.Len(t, s, 3)
	for i := 0; i < s[0]; i++ {
		for j := 0; j < s[1]; j++ {
			for k := 0; k < s[2]; k++ {
				assert.Equal(t, tn.Loc([]int{i, j, k}), v)
				v += 1
			}
		}
	}
}
