// internal/realtime/upgrader.go
package realtime

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

const (
	wsPongWait   = 60 * time.Second
	wsPingPeriod = 45 * time.Second
	wsWriteWait  = 10 * time.Second
	wsMaxMsgSize = 512 * 1024
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}
