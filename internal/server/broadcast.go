package server

import (
	"fmt"
	"sync"

	kiteconnect "github.com/zerodha/gokiteconnect/v4/models"
)

var (
	subscribers     = make(map[chan kiteconnect.Tick]struct{})
	subscribersLock sync.Mutex
)

// Register adds a new subscriber channel to broadcast ticks.
func Register() chan kiteconnect.Tick {
	ch := make(chan kiteconnect.Tick, 100)
	subscribersLock.Lock()
	subscribers[ch] = struct{}{}
	subscribersLock.Unlock()
	return ch
}

// Unregister removes and closes a subscriber channel.
func Unregister(ch chan kiteconnect.Tick) {
	subscribersLock.Lock()
	delete(subscribers, ch)
	close(ch)
	subscribersLock.Unlock()
}

// BroadcastTick sends the tick to all subscribers.
func BroadcastTick(tick kiteconnect.Tick) {
	subscribersLock.Lock()
	defer subscribersLock.Unlock()

	for ch := range subscribers {
		select {
		case ch <- tick:
		default:
			fmt.Println("❗ Dropping tick: slow client")
		}
	}
}
