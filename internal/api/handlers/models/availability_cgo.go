//go:build cgo

package models

// Built with cgo, so the ONNX runtime bindings can be linked.
const (
	inferenceAvailable = true
	inferenceNote      = "ONNX runtime bindings are compiled in (cgo enabled)."
)
