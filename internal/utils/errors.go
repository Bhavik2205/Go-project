package utils

import (
	"fmt"

	"go.uber.org/zap"
)

type CustomError struct {
	Code    int
	Message string
	Err     error
	Meta    map[string]interface{}
}

func (e *CustomError) Error() string {
	return fmt.Sprintf("Code %d: %s | Details: %v", e.Code, e.Message, e.Err)
}

func (e *CustomError) Unwrap() error {
	return e.Err
}

func WrapError(code int, message string, err error, meta ...map[string]interface{}) *CustomError {
	customErr := &CustomError{
		Code:    code,
		Message: message,
		Err:     err,
	}
	if len(meta) > 0 {
		customErr.Meta = meta[0]
	}

	logFields := []zap.Field{
		zap.Int("code", code),
		zap.String("message", message),
		zap.Error(err),
	}
	if customErr.Meta != nil {
		for k, v := range customErr.Meta {
			logFields = append(logFields, zap.Any(k, v))
		}
	}

	zap.L().Error("An error occurred", logFields...)
	return customErr
}
