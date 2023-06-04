package examples

import (
	"dexianta/tgnn/core"
	"dexianta/tgnn/data"
	"fmt"
	"math"
	"testing"
)

func TestBasicMnist(t *testing.T) {
	weights := core.Randn(784, 10).DivS(math.Sqrt(784))
	bias := core.Zeros(10)

	train, _ := data.MnistLoader()
	trainX, trainY := train.Tensors()
	//testX, testY := test.Tensors()

	model := func(input core.Tensor) core.Tensor {
		matmul := input.Matmul(weights).Add(bias)
		return core.LogSoftmax(matmul, 1) // the max of log_softmax is 0
	}

	lossFunc := func(input, target core.Tensor) *core.V {
		var pos = make([]core.Pos, input.Shape[0])
		for i := range pos {
			pos[i] = make([]int, 2)
			pos[i][0] = i
			pos[i][1] = int(target.GetV(core.Pos{i}).Data)
		}

		return core.Mean(input.GetBatchV(pos)).Mul(core.Vx(-1))
	}

	preds := model(trainX.Slice(core.S{0, 64}))
	target := trainY.Slice(core.S{0, 64})
	loss := lossFunc(preds, target)
  loss.Backward()
	fmt.Println(loss)
}
