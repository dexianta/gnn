package nn

import (
	"dexianta/tgnn/common"
	"math/rand"
)

// Neuron is smallest unit in a neural network, which can be represented as follows
// `activation_function(sum(x1w1, x2w2, xnwn) + b)`
// common activation_function are:
// - sigmod
// - relu
// - etc
type Neuron struct {
	input int
	ws    []*common.V
	b     *common.V
}

func NewNeuron(ninput int) Neuron {
	n := Neuron{
		input: ninput,
		ws:    make([]*common.V, ninput),
		b:     common.NewVar(0.),
	}

	// initialize random weight value (-1, 1)
	for i := range n.ws {
		n.ws[i] = common.NewVar(rand.Float64()*2. - 1.)
	}
	return n
}

type Layer struct {
	neurons []Neuron
}

func NewLayer(inputSize, outputSize int) Layer {
	var neurons = make([]Neuron, outputSize)
	for i := 0; i < outputSize; i++ {
		neurons[i] = NewNeuron(inputSize)
	}

	return Layer{
		neurons: neurons,
	}
}
