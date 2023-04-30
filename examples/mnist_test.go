package examples

import (
	"dexianta/tgnn/core"
	"math"
	"testing"
)

func TestMinstBasic(t *testing.T) {
	weights := core.Randn(784, 10).Div(math.Sqrt(784)) // Xavier initialization
	bias := core.Zeros(10)

  model := func(input Tensor) Tensor {
    
  }
}
