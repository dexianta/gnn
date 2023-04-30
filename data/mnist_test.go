package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadMnist(t *testing.T) {
	train, test := MnistLoader()
	assert.Equal(t, len(train.Images), 60000)
	assert.Equal(t, len(train.Images[0]), 784)
	assert.Equal(t, len(train.Labels), 60000)

	assert.Equal(t, len(test.Images), 10000)
	assert.Equal(t, len(test.Images[0]), 784)
	assert.Equal(t, len(test.Labels), 10000)
}
