package examples

import (
	"dexianta/tgnn/core"
	"dexianta/tgnn/data"
	"fmt"
	"math"
	"testing"
)

// this example roughly follow the example given here:
// https://pytorch.org/tutorials/beginner/nn_tutorial.html
func TestBasicMnist(t *testing.T) {
	weights := core.Randn(784, 10).DivS(math.Sqrt(784))
	bias := core.Zeros(10)

	train, _ := data.MnistLoader()
	trainX, trainY := train.Tensors()
	//testX, testY := test.Tensors()

	// the input batch has the size of (64 (batch size), 10)
	// after the log softmax, the highest value will be approaching 0
	model := func(input core.Tensor) core.Tensor {
		matmul := input.Matmul(weights).Add(bias)
		return core.LogSoftmax(matmul, 1) // the max of log_softmax is 0
	}

	// input shape: [batch size, 10]
	// target shape: [batch size], each element is 0 to 9
	// why this loss function works:
	// for one input:
	// [-20, -30, -40, -24, ... , -10] (size 10, for 10 digits)
	// target:
	//    5
	// mean([input[idx][target[i]] for i in range(batch_size)]) * -1
	// we are optimizing so that the probability of input[idx][target[idx]]
	// to be the highest, the reason for multiply by -1 is log_softmax has all number to be negative
	lossFunc := func(input, target core.Tensor) *core.V {
		var pos = make([]core.Pos, input.Shape[0])
		for i := range pos {
			pos[i] = make([]int, 2) // input and target are both dim 2 tensor
			pos[i][0] = i
			pos[i][1] = int(target.GetV(core.Pos{i}).Data)
		}

		return core.Mean(input.GetVs(pos)).Mul(core.Vx(-1))
	}

	//TODO:
	//accuracy := func(out, yb core.Tensor) float64 {
	//	return 0.
	//}

	preds := model(trainX.Slice(core.S{0, 64}))
	target := trainY.Slice(core.S{0, 64})
	loss := lossFunc(preds, target)
	fmt.Println(loss)
}
