package nn

import (
	"dexianta/tgnn/autograd"
	"math/rand"
)

type Neuron struct {
	input int
	ws    []*autograd.Var
	b     *autograd.Var
}

func NewNeuron(ninput int) *Neuron {
	n := &Neuron{
		input: ninput,
		ws:    make([]*autograd.Var, ninput),
		b:     autograd.NewVar(0.),
	}

	for i := range n.ws {
		n.ws[i] = autograd.NewVar(rand.Float64()*2. - 1.)
	}
  return n
}
