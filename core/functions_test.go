package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSoftmax(t *testing.T) {
	df := Ones(4, 3, 2)
	ret0 := Softmax(df, 0)
	ret1 := Softmax(df, 1)
	ret2 := Softmax(df, 2)

	expected0 := All(0.25, []int{4, 3, 2})
	expected1 := All(1./3., []int{4, 3, 2})
	expected2 := All(0.5, []int{4, 3, 2})
	assert.True(t, expected0.Equal(ret0))
	assert.True(t, expected1.Equal(ret1))
	assert.True(t, expected2.Equal(ret2))

}
