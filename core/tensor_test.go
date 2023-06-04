package core

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBroadCastable(t *testing.T) {
	a := Ones(2, 3, 3)
	b := Ones(1)
	c := Ones(3, 3)

	assert.True(t, canBroadcast(a, b))
	assert.True(t, canBroadcast(a, c))
	assert.True(t, canBroadcast(c, a))
	assert.False(t, canBroadcast(a, Ones(2, 3)))
}

func TestRand(t *testing.T) {
	Randn(3, 3).PrintData()
}

func TestTensorInit(t *testing.T) {
	tn := NewTensor(d3{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}})

	s := tn.Shape
	v := 1.0
	assert.Len(t, s, 3)
	for i := 0; i < s[0]; i++ {
		for j := 0; j < s[1]; j++ {
			for k := 0; k < s[2]; k++ {
				assert.Equal(t, tn.Loc([]int{i, j, k}), v)
				v += 1
			}
		}
	}
}

func TestShapeIter(t *testing.T) {
	a := Shape{3, 4}

	iter := a.Iter()
	for iter.Next() {
		pos := iter.Step()
		assert.Equal(t, iter.idx-1, toIndex(pos, a))
	}

	assert.Equal(t, 12, iter.idx)
}

func TestSlice(t *testing.T) {
	df := NewTensor(d2{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})

	s1 := df.Slice(S{0, 2})
	expected1 := NewTensor(d2{{1, 2, 3}, {4, 5, 6}})
	assert.True(t, s1.Equal(expected1))

	s2 := df.Slice(S{1, 2})
	expected2 := NewTensor(d2{{4, 5, 6}})
	assert.True(t, s2.Equal(expected2))

	s3 := df.Slice(S{0, 2}, S{1, 3})
	expected3 := NewTensor(d2{{2, 3}, {5, 6}})
	assert.True(t, s3.Equal(expected3))
}

func TestInRange(t *testing.T) {
	// {3,3,3} slice {1:2}
	shape := Shape{3, 3, 3}
	sl := [][2]int{{0, 2}}
	verbose := toVerboseSlice(sl, shape)
	for i := 0; i < 27; i++ {
		ok, pos := inRange(i, verbose, shape)
		fmt.Println(toPos(i, shape), ok, pos)
	}
}

func TestAdd(t *testing.T) {
	a := Ones(2, 3, 3)
	b := Ones(3, 3)

	c := a.Add(b)
	d := Ones(2, 3, 3).Add(Ones(1))

	assert.True(t, c.Equal(d))
	for _, v := range c.data {
		assert.Equal(t, v.Data, 2.0)
	}

	a = NewTensor(d2{{1, 2, 3}, {4, 5, 6}})
	b = NewTensor(d1{1, 2, 3})
	c = a.Add(b)
	d = NewTensor(d2{{2, 4, 6}, {5, 7, 9}})
	assert.True(t, c.Equal(d))
}

func TestMatMul(t *testing.T) {
	t.Run("", func(t *testing.T) {
		a := NewTensor(d2{{1, 2}, {3, 4}})
		b := NewTensor(d2{{3, 4, 5}, {6, 7, 8}})
		expected := NewTensor(d2{{15, 18, 21}, {33, 40, 47}})

		c := a.Matmul(b)
		assert.Equal(t, c.Shape, Shape{2, 3})
		assert.True(t, c.Equal(expected))
	})

	t.Run("", func(t *testing.T) {
		a := Ones(2, 3, 4)
		b := Ones(2, 4, 5)

		c := a.Matmul(b)
		assert.Equal(t, c.Shape, Shape{2, 3, 5})
		assert.True(t, c.Equal(Ones(2, 3, 5).MulS(4)))
	})
}
