package contracts

import "time"

const APIVersionV1 = "v1"

type Meta struct {
	RequestID  string      `json:"requestId"`
	ServerTime time.Time   `json:"serverTime"`
	Version    string      `json:"version"`
	Pagination *Pagination `json:"pagination,omitempty"`
}

type Pagination struct {
	Limit      int    `json:"limit"`
	Cursor     string `json:"cursor,omitempty"`
	NextCursor string `json:"nextCursor,omitempty"`
	HasMore    bool   `json:"hasMore"`
}

type SuccessResponse[T any] struct {
	Data T    `json:"data"`
	Meta Meta `json:"meta"`
}

type ErrorResponse struct {
	Error APIError `json:"error"`
	Meta  Meta     `json:"meta"`
}

type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details any    `json:"details,omitempty"`
}

func NewMeta(requestID string, serverTime time.Time) Meta {
	return Meta{
		RequestID:  requestID,
		ServerTime: serverTime,
		Version:    APIVersionV1,
	}
}

func NewSuccess[T any](requestID string, data T) SuccessResponse[T] {
	return SuccessResponse[T]{
		Data: data,
		Meta: NewMeta(requestID, time.Now()),
	}
}

func NewError(requestID, code, message string, details any) ErrorResponse {
	return ErrorResponse{
		Error: APIError{
			Code:    code,
			Message: message,
			Details: details,
		},
		Meta: NewMeta(requestID, time.Now()),
	}
}
