package strategy

import (
	"fmt"
	"sort"
	"sync"
)

// The registry is how strategies get wired in. Adding a new strategy is a
// single self-contained file:
//
//	package strategy
//
//	func init() {
//	    Register("my_strategy", func() Strategy { return &MyStrategy{} })
//	}
//
//	type MyStrategy struct{ base; ... }
//	func (s *MyStrategy) Name() string { return "my_strategy" }
//	func (s *MyStrategy) Init(p Params) error { ... }
//	func (s *MyStrategy) OnCandle(f *Frame, pos *Position) Signal { ... }
//
// No other file needs to change. Enable it by name in configs/strategy.yaml
// and the engine will instantiate it per (instrument, interval).

var (
	registryMu sync.RWMutex
	registry   = make(map[string]Factory)
)

// Register adds a strategy factory under name. It panics on a duplicate or
// empty name: both are programmer errors that must fail at startup, not at
// 09:15 on a live feed.
func Register(name string, f Factory) {
	if name == "" {
		panic("strategy: Register called with empty name")
	}
	if f == nil {
		panic("strategy: Register called with nil factory for " + name)
	}

	registryMu.Lock()
	defer registryMu.Unlock()

	if _, dup := registry[name]; dup {
		panic("strategy: duplicate registration for " + name)
	}
	registry[name] = f
}

// New builds and initialises a registered strategy.
func New(name string, p Params) (Strategy, error) {
	registryMu.RLock()
	f, ok := registry[name]
	registryMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("strategy %q is not registered (known: %v)", name, Names())
	}

	s := f()
	if s == nil {
		return nil, fmt.Errorf("strategy %q: factory returned nil", name)
	}
	if err := s.Init(p); err != nil {
		return nil, fmt.Errorf("strategy %q: init: %w", name, err)
	}
	if err := validate(s); err != nil {
		return nil, err
	}
	if got := s.Name(); got != name {
		return nil, fmt.Errorf("strategy registered as %q reports Name() = %q", name, got)
	}
	return s, nil
}

// Names lists every registered strategy, sorted for stable logs and tests.
func Names() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()

	out := make([]string, 0, len(registry))
	for n := range registry {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// IsRegistered reports whether name is known. Used to validate config before
// the engine starts.
func IsRegistered(name string) bool {
	registryMu.RLock()
	defer registryMu.RUnlock()
	_, ok := registry[name]
	return ok
}
