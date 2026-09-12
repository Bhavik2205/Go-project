//go:build !cgo

package models

// Built without cgo. github.com/yalue/onnxruntime_go needs cgo, so sentiment
// inference cannot run in this binary and news articles will stay unscored.
const (
	inferenceAvailable = false
	inferenceNote      = "Built without cgo: ONNX inference is unavailable, so news sentiment will not be scored. Rebuild with CGO_ENABLED=1 and the ONNX runtime present."
)
