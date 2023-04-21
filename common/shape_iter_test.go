package common

import (
	"fmt"
	"testing"
)

func TestShapeIter(t *testing.T) {
	iter := newShapeIter([]int{0, 0}, []int{3, 3})
	fmt.Println(iter)
}
