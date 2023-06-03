package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVal(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := Vx(3)
		d := a.Add(a)
		d.backward(1)
		assert.Equal(t, a.Grad, 2.0)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(3)
		b := a.Neg()
		b.backward(1)
		assert.Equal(t, a.Grad, -1.)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(1)
		b := Vx(2)
		d := a.Sub(b) // a - (b * -1)
		d.backward(1)
		assert.Equal(t, a.Grad, 1.0)
		assert.Equal(t, b.Grad, -1.0)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(3)
		b := Vx(5)
		c := Vx(7)
		d := Vx(9)
		f := (a.Mul(b.Sub(c)).Pow(2)).Add(d) // (a * (b - c))^2 + d

		f.backward(1)

		// verified with pytorch
		assert.Equal(t, a.Grad, 24.0) // 2(a * (b-c))(b - c) = 2 * (3 * (-2)) * -2 = 24
		assert.Equal(t, b.Grad, -36.)
		assert.Equal(t, c.Grad, 36.) // 2(a * (b-c))*a*(-1) = 2 * (3 * -2)*2*-1 = 36
		assert.Equal(t, d.Grad, 1.)

		f.Zerograd()
		assert.Equal(t, a.Grad, 0.)
		assert.Equal(t, b.Grad, 0.)
		assert.Equal(t, c.Grad, 0.)
		assert.Equal(t, d.Grad, 0.)
	})

	t.Run("relu", func(t *testing.T) {
		a := Vx(3)
		b := a.ReLu()
		b.backward(1)

		assert.Equal(t, b.Data, 3.)
		assert.Equal(t, a.Grad, 1.)

		a = Vx(-3)
		b = a.ReLu()
		b.backward(1)

		assert.Equal(t, b.Data, 0.)
		assert.Equal(t, a.Grad, 0.)
	})
}
