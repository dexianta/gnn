package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLogSoftmax(t *testing.T) {
	a := Ones(4, 3, 2)
	ret0 := LogSoftmax(a, 0)
	ret1 := LogSoftmax(a, 1)
	ret2 := LogSoftmax(a, 2)

	expected0 := All(-1.3863, Shape{4, 3, 2})
	expected1 := All(-1.0986, Shape{4, 3, 2})
	expected2 := All(-0.6931, Shape{4, 3, 2})

	assert.True(t, ret0.Equal(expected0))
	assert.True(t, ret1.Equal(expected1))
	assert.True(t, ret2.Equal(expected2))
}

func TestSoftmax(t *testing.T) {
	df := Ones(4, 3, 2)
	ret0 := Softmax(df, 0)
	ret1 := Softmax(df, 1)
	ret2 := Softmax(df, 2)

	expected0 := All(0.25, []int{4, 3, 2})
	expected1 := All(1./3., []int{4, 3, 2})
	expected2 := All(0.5, []int{4, 3, 2})
	assert.True(t, expected0.Equal(ret0))
	assert.True(t, expected1.Equal(ret1))
	assert.True(t, expected2.Equal(ret2))
}

func TestLogSoftmaxBackward(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := NewTensor(d1{1., 2., 3., 4.})
		b := LogSoftmax(a, 0)
		b.GetV(Pos{0}).Backward()
		assert.Nil(t,
			EqualFloatArray(
				a.Grad(),
				[]float64{0.9679, -0.0871, -0.2369, -0.6439},
				0.001),
		)
	})

	t.Run("", func(t *testing.T) {
		a := NewTensor(d2{{1, 2, 3}, {4, 5, 6}})
		b := LogSoftmax(a, 0)
		c := LogSoftmax(a, 1)
		b.GetV(Pos{0, 0}).Backward()
		c.GetV(Pos{1, 1}).Backward()

		assert.Nil(t,
			EqualFloatArray(
				a.Grad(),
				[]float64{0.9526, 0, 0, -1.0426, 0.7553, -0.6652},
				0.01,
			),
		)
	})
}

func TestSoftmaxBackward(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := NewTensor(d1{1., 2., 3., 4.})
		b := Softmax(a, 0)
		b.GetV(Pos{0}).Backward()
		assert.Nil(t,
			EqualFloatArray(
				a.Grad(),
				[]float64{0.031, -0.0028, -0.0076, -0.0206},
				0.01),
		)
	})

	t.Run("", func(t *testing.T) {
		a := NewTensor(d2{{1, 2, 3}, {4, 5, 6}})
		b := Softmax(a, 0)
		c := Softmax(a, 1)
		b.GetV(Pos{0, 0}).Backward()
		c.GetV(Pos{1, 1}).Backward()

		assert.Nil(t,
			EqualFloatArray(
				a.Grad(),
				[]float64{0.0452, 0, 0, -0.0672, 0.1848, -0.1628},
				0.01,
			),
		)
	})
}
