package data

import (
	"archive/zip"
	"dexianta/tgnn/util"
	"encoding/binary"
	"fmt"
	"io"
	"io/fs"
)

type MNIST struct {
	Images [][]uint8
	Labels []uint8
}

func readSet(imgFile, lblFile fs.File) (ret MNIST) {
	imgBytes, _ := io.ReadAll(imgFile)
	lblBytes, _ := io.ReadAll(lblFile)

	if len(imgBytes) < 16 {
		util.Panic(fmt.Errorf("invalid images file"))
	}

	if len(lblBytes) < 8 {
		util.Panic(fmt.Errorf("invalid labels file"))
	}

	if binary.BigEndian.Uint32(imgBytes[0:4]) != 0x00000803 {
		panic("invalid image magic number")
	}
	if binary.BigEndian.Uint32(lblBytes[0:4]) != 0x00000801 {
		panic("invalid label magic number")
	}

	imgCnt := binary.BigEndian.Uint32(imgBytes[4:8])
	lblCnt := binary.BigEndian.Uint32(lblBytes[4:8])

	if imgCnt != lblCnt {
		panic("image and label counts doesn't match")
	}

	row := int(binary.BigEndian.Uint32(imgBytes[8:12]))
	cols := int(binary.BigEndian.Uint32(imgBytes[12:16]))

	images := make([][]uint8, imgCnt)
	labels := make([]uint8, lblCnt)

	for i := 0; i < int(imgCnt); i++ {
		image := make([]uint8, row*cols)
		for j := 0; j < row*cols; j++ {
			image[j] = imgBytes[16+i*row*cols+j]
		}
		images[i] = image
		labels[i] = lblBytes[8+i]
	}

	ret.Images = images
	ret.Labels = labels
	return
}

func MnistLoader() (trainSet, testSet MNIST) {
	zipReader, err := zip.OpenReader("../resources/mnist.zip")
	util.Panic(err)
	defer zipReader.Close()

	trainX, _ := zipReader.Open("train-images.idx3-ubyte")
	trainY, _ := zipReader.Open("train-labels.idx1-ubyte")
	testX, _ := zipReader.Open("t10k-images.idx3-ubyte")
	testY, _ := zipReader.Open("t10k-labels.idx1-ubyte")

	trainSet = readSet(trainX, trainY)
	testSet = readSet(testX, testY)
	return
}
