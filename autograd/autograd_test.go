package autograd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVal(t *testing.T) {
	a := NewVal(3)

	d := a.Add(a)

	d.Backward(1)

	assert.Equal(t, a.grad, 2.0)
}
