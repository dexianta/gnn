package examples

import (
	"dexianta/tgnn/core"
	"dexianta/tgnn/data"
	"fmt"
	"math"
	"testing"
)

func TestMinstBasic(t *testing.T) {
	weights := core.Randn(784, 10).DivS(math.Sqrt(784)) // Xavier initialization
	bias := core.Zeros(10)

	train, test := data.MnistLoader()
	trainX, trainY := train.Tensors()
	testX, testY := test.Tensors()

	model := func(input core.Tensor) core.Tensor {
		return input.Matmul(weights).Add(bias)
	}

	fmt.Println(trainY.Shape, testX.Shape, testY.Shape)
	model(trainX.Slice(core.S{0, 64}))
}
