package contracts

import "time"

const (
	WSMessageSubscribe   = "SUBSCRIBE"
	WSMessageUnsubscribe = "UNSUBSCRIBE"
	WSMessageError       = "ERROR"

	WSTopicMarketTicks      = "market.ticks"
	WSTopicMarketCandles    = "market.candles"
	WSTopicMarketIndicators = "market.indicators"
	WSTopicMarketHeatmap    = "market.heatmap"
	WSTopicOrders           = "orders"
	WSTopicPositions        = "positions"
	WSTopicAlerts           = "alerts"
	WSTopicStrategies       = "strategies"
	WSTopicModels           = "models"
	WSTopicBacktests        = "backtests"
)

type WSSubscribeRequest struct {
	Type      string            `json:"type"`
	RequestID string            `json:"requestId,omitempty"`
	Topics    []string          `json:"topics"`
	Filters   map[string]any    `json:"filters,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type WSEvent[T any] struct {
	Type       string    `json:"type"`
	Topic      string    `json:"topic"`
	EventID    string    `json:"eventId"`
	ServerTime time.Time `json:"serverTime"`
	Data       T         `json:"data"`
}

type WSErrorEvent struct {
	Type       string    `json:"type"`
	Topic      string    `json:"topic"`
	EventID    string    `json:"eventId"`
	ServerTime time.Time `json:"serverTime"`
	Error      APIError  `json:"error"`
}

func NewWSEvent[T any](eventType, topic, eventID string, data T) WSEvent[T] {
	return WSEvent[T]{
		Type:       eventType,
		Topic:      topic,
		EventID:    eventID,
		ServerTime: time.Now(),
		Data:       data,
	}
}

func NewWSError(eventID, topic, code, message string, details any) WSErrorEvent {
	return WSErrorEvent{
		Type:       WSMessageError,
		Topic:      topic,
		EventID:    eventID,
		ServerTime: time.Now(),
		Error: APIError{
			Code:    code,
			Message: message,
			Details: details,
		},
	}
}
