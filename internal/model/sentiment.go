package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"sync"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

type TokenizedOutput struct {
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask"`
}

var ortInitOnce sync.Once
var ortInitErr error

func softmax(logits []float32) []float32 {
	max := logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	expSum := float32(0.0)
	for i := range logits {
		logits[i] = float32(math.Exp(float64(logits[i] - max))) // prevent overflow
		expSum += logits[i]
	}
	for i := range logits {
		logits[i] /= expSum
	}
	return logits
}

// initializeORT handles the one-time initialization of ONNX Runtime.
func initializeORT() error {
	ortInitOnce.Do(func() {
		// Consider making this path configurable
		dllPath := os.Getenv("ONNX_DLL_PATH")
		if dllPath == "" {
			fmt.Println("Please set the ONNX_DLL_PATH environment variable.")
			return
		}
		onnxruntime.SetSharedLibraryPath(dllPath)
		ortInitErr = onnxruntime.InitializeEnvironment()
		if ortInitErr != nil {
			ortInitErr = fmt.Errorf("error initializing ONNX Runtime environment: %w", ortInitErr)
		}
	})
	return ortInitErr
}

func AnalyzeSentiment(text string) (string, float32, error) {
	if err := initializeORT(); err != nil {
		return "", 0, err // Return the initialization error directly
	}

	// Tokenization (ensure your python script handles errors and prints JSON reliably)
	cmd := exec.Command("python", "scripts/tokenize_text.py", text)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", 0, fmt.Errorf("tokenization script failed: %v. Output: %s", err, string(out))
	}

	lines := bytes.Split(out, []byte("\n"))
	var jsonLine []byte
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(string(lines[i]))
		if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
			jsonLine = lines[i]
			break
		}
	}
	if jsonLine == nil {
		return "", 0, fmt.Errorf("no JSON found in tokenizer output. Full output: %s", string(out))
	}

	var input TokenizedOutput
	if err := json.Unmarshal(jsonLine, &input); err != nil {
		return "", 0, fmt.Errorf("failed to parse tokenized JSON: %v. JSON line: %s", err, string(jsonLine))
	}

	// Define tensor shapes
	// Ensure your tokenizer pads/truncates inputs to this fixed sequence length
	const sequenceLength = 128
	if len(input.InputIDs) != sequenceLength || len(input.AttentionMask) != sequenceLength {
		return "", 0, fmt.Errorf("tokenized input length mismatch: expected %d, got %d for input_ids and %d for attention_mask. Ensure tokenizer pads/truncates",
			sequenceLength, len(input.InputIDs), len(input.AttentionMask))
	}
	shape := onnxruntime.Shape{1, sequenceLength}

	// Create input tensors (type int64)
	inputIDsTensor, err := onnxruntime.NewTensor(shape, input.InputIDs)
	if err != nil {
		return "", 0, fmt.Errorf("input_ids tensor error: %w", err)
	}
	attentionMaskTensor, err := onnxruntime.NewTensor(shape, input.AttentionMask)
	if err != nil {
		return "", 0, fmt.Errorf("attention_mask tensor error: %w", err)
	}

	// Create empty output tensor (type float32, as expected by the model's "logits" output)
	outputShape := onnxruntime.Shape{1, 3} // [batch_size, num_sentiment_classes]
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create output tensor: %w", err)
	}

	modelPath := "D:/troject/go-project/models/sentiment_optimized.onnx" // Consider making this configurable

	// Use NewAdvancedSession (function call with parentheses)
	session, err := onnxruntime.NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"}, // Model's input names
		[]string{"logits"},                      // Model's output name
		[]onnxruntime.Value{inputIDsTensor, attentionMaskTensor}, // Input tensors
		[]onnxruntime.Value{outputTensor},                        // Output tensor(s)
		nil,                                                      // SessionConfig (can be nil for default configuration)
	)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create ONNX AdvancedSession: %w", err)
	}
	defer session.Destroy()

	// Run inference
	// The Run method for AdvancedSession doesn't take arguments,
	// as inputs/outputs were bound during NewAdvancedSession.
	if err := session.Run(); err != nil {
		return "", 0, fmt.Errorf("ONNX inference run failed: %w", err)
	}

	// Process the output
	logits := outputTensor.GetData() // This will be []float32
	if len(logits) != 3 {
		return "", 0, fmt.Errorf("unexpected logits length: got %d, expected 3. Logits: %v", len(logits), logits)
	}

	labels := []string{"negative", "neutral", "positive"}
	probabilities := softmax(logits)
	maxIdx := 0
	maxVal := probabilities[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}

	if maxIdx < 0 || maxIdx >= len(labels) { // Should not happen if len(logits) == 3
		return "", 0, fmt.Errorf("internal error: maxIdx %d is out of bounds for labels", maxIdx)
	}

	return labels[maxIdx], maxVal, nil
}
