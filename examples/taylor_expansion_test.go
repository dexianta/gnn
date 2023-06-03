package examples

import (
	"dexianta/tgnn/core"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestTaylorWithV(t *testing.T) {
	// e^x = 1 + x + x^2/2! + x^3/3! + ... + x^n / n!
	// y = 1 + x + a*x^2 + b*x^3 + c*x^4 + d*x^5
	a := core.Vx(rand.NormFloat64()) // true val: 0.5
	b := core.Vx(rand.NormFloat64()) // true val: 0.1666..
	c := core.Vx(rand.NormFloat64()) // true val: 0.0416666..
	d := core.Vx(rand.NormFloat64()) // true val: 0.0083333..

	taylor5th := func(x *core.V) *core.V {
		return core.Vx(1).
			Add(x).
			Add(core.Vx(math.Pow(x.Data, 2)).Mul(a)).
			Add(core.Vx(math.Pow(x.Data, 3)).Mul(b)).
			Add(core.Vx(math.Pow(x.Data, 4)).Mul(c)).
			Add(core.Vx(math.Pow(x.Data, 5)).Mul(d))
	}

	lossFunc := func() (*core.V, float64) {
		// compute the loss
		r := core.Range(-3, 3, 0.5)
		x := core.Map(r, func(x float64) *core.V {
			return core.Vx(x)
		})
		y := core.Map(r, func(x float64) *core.V {
			return core.Vx(math.Exp(x))
		})

		var tmp []*core.V
		for i := range y {
			tmp = append(tmp, (y[i].Sub(taylor5th(x[i]))).Pow(2))
		}

		// compute MSE
		var mse float64
		for i := range y {
			err := taylor5th(core.Vx(x[i].Data)).Sub(y[i]).Data
			mse += err * err
		}

		return core.Sum(tmp), mse / float64(len(y))
	}

	//learningRate := 0.001
	mop := core.NewMomentumOptim([]*core.V{a, b, c, d}, 0.00001, 0.9)
	for i := 0; i < 1000; i++ {
		loss, mse := lossFunc()
		loss.Backward()
		mop.Step()
		if i%10 == 0 {
			fmt.Printf("==== (%d) ===\nloss: %f, mse: %f, (a, b, c, d): (%f, %f, %f, %f)\n",
				i, loss.Data, mse, a.Data, b.Data, c.Data, d.Data)
		}
		mop.ZeroGrad()
	}
}
