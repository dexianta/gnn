package common

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVal(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := Vx(3)
		d := a.Add(a)
		d.backward(1)
		assert.Equal(t, a.grad, 2.0)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(3)
		b := a.Neg()
		b.backward(1)
		assert.Equal(t, a.grad, -1.)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(1)
		b := Vx(2)
		d := a.Sub(b) // a - (b * -1)
		d.backward(1)
		assert.Equal(t, a.grad, 1.0)
		assert.Equal(t, b.grad, -1.0)
	})

	t.Run("", func(t *testing.T) {
		a := Vx(3)
		b := Vx(5)
		c := Vx(7)
		d := Vx(9)
		f := (a.Mul(b.Sub(c)).Pow(2)).Add(d) // (a * (b - c))^2 + d

		f.backward(1)

		// verified with pytorch
		assert.Equal(t, a.grad, 24.0) // 2(a * (b-c))(b - c) = 2 * (3 * (-2)) * -2 = 24
		assert.Equal(t, b.grad, -36.)
		assert.Equal(t, c.grad, 36.) // 2(a * (b-c))*a*(-1) = 2 * (3 * -2)*2*-1 = 36
		assert.Equal(t, d.grad, 1.)

		f.Zerograd()
		assert.Equal(t, a.grad, 0.)
		assert.Equal(t, b.grad, 0.)
		assert.Equal(t, c.grad, 0.)
		assert.Equal(t, d.grad, 0.)
	})

	t.Run("relu", func(t *testing.T) {
		a := Vx(3)
		b := a.ReLu()
		b.backward(1)

		assert.Equal(t, b.data, 3.)
		assert.Equal(t, a.grad, 1.)

		a = Vx(-3)
		b = a.ReLu()
		b.backward(1)

		assert.Equal(t, b.data, 0.)
		assert.Equal(t, a.grad, 0.)
	})
}

func TestTaylor(t *testing.T) {
	// e^x = 1 + x + x^2/2! + x^3/3! + ... + x^n / n!
	// y = 1 + x + a*x^2 + b*x^3 + c*x^4
	a := Vx(rand.NormFloat64()) // true val: 0.5
	b := Vx(rand.NormFloat64()) // true val: 0.1666..
	c := Vx(rand.NormFloat64()) // true val: 0.0416666..
	d := Vx(rand.NormFloat64()) // true val: 0.0083333..

	taylor5th := func(x *V) *V {
		return Vx(1).
			Add(x).
			Add(Vx(math.Pow(x.data, 2)).Mul(a)).
			Add(Vx(math.Pow(x.data, 3)).Mul(b)).
			Add(Vx(math.Pow(x.data, 4)).Mul(c)).
			Add(Vx(math.Pow(x.data, 5)).Mul(d))
	}

	lossFunc := func() (*V, float64) {
		// compute the loss
		r := Range(-3, 3, 0.5)
		x := Map(r, func(x float64) *V {
			return Vx(x)
		})
		y := Map(r, func(x float64) *V {
			return Vx(math.Exp(x))
		})

		var tmp []*V
		for i := range y {
			tmp = append(tmp, (y[i].Sub(taylor5th(x[i]))).Pow(2))
		}

		// compute MSE
		var mse float64
		for i := range y {
			err := taylor5th(Vx(x[i].data)).Sub(y[i]).data
			mse += err * err
		}

		return Sum(tmp), mse / float64(len(y))
	}

	//learningRate := 0.001
	mop := NewMomentumOptim([]*V{a, b, c, d}, 0.00001, 0.9)
	for i := 0; i < 1000; i++ {
		loss, mse := lossFunc()
		loss.Backward()
		mop.Step()
		if i%10 == 0 {
			fmt.Printf("==== (%d) ===\nloss: %f, mse: %f, (a, b, c, d): (%f, %f, %f, %f)\n",
				i, loss.data, mse, a.data, b.data, c.data, d.data)
		}
		mop.ZeroGrad()
	}
}
