package autograd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVal(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := NewVal(3)
		d := a.Add(a)
		d.Backward(1)
		assert.Equal(t, a.grad, 2.0)
	})

	t.Run("", func(t *testing.T) {
		a := NewVal(3)
		b := a.Neg()
		b.Backward(1)
		assert.Equal(t, a.grad, -1.)
	})

	t.Run("", func(t *testing.T) {
		a := NewVal(1)
		b := NewVal(2)
		d := a.Sub(b) // a - (b * -1)
		d.Backward(1)
		assert.Equal(t, a.grad, 1.0)
		assert.Equal(t, b.grad, -1.0)
	})

	t.Run("", func(t *testing.T) {
		a := NewVal(3)
		b := NewVal(5)
		c := NewVal(7)
		d := NewVal(9)
		f := (a.Mul(b.Sub(c)).Pow(2)).Add(d) // (a * (b - c))^2 + d

		f.Backward(1)

		// verified with pytorch
		assert.Equal(t, a.grad, 24.0) // 2(a * (b-c))(b - c) = 2 * (3 * (-2)) * -2 = 24
		assert.Equal(t, b.grad, -36.)
		assert.Equal(t, c.grad, 36.) // 2(a * (b-c))*a*(-1) = 2 * (3 * -2)*2*-1 = 36
		assert.Equal(t, d.grad, 1.)
	})
}
