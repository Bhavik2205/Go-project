// internal/utils/symbols.go
package utils

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type SimulatedInstrument struct {
	Token          uint32  `yaml:"token"`
	Symbol         string  `yaml:"symbol"`
	Exchange       string  `yaml:"exchange"`
	InstrumentType string  `yaml:"instrument_type"`
	Name           string  `yaml:"name"`
	Segment        string  `yaml:"segment"`
	TickSize       float64 `yaml:"tick_size"`
	LotSize        uint32  `yaml:"lot_size"`
}

type SymbolsConfig struct {
	SimulationInstruments []SimulatedInstrument `yaml:"simulation_instruments"`
	LiveSymbols           []string              `yaml:"live_symbols"`
}

func LoadSymbolsConfig(path string) (*SymbolsConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read symbols config file %s: %w", path, err)
	}
	var cfg SymbolsConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal symbols config: %w", err)
	}
	return &cfg, nil
}
