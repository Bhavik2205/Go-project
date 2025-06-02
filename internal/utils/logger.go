// internal/utils/logger.go
package utils

import (
	"log"
	"os"
	"strings"
)

// InitLogger initializes the global logger based on config.
func InitLogger(level, output string) {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	if output == "file" {
		file, err := os.OpenFile("bot.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			log.Fatalf("Failed to open log file: %v", err)
		}
		log.SetOutput(file)
	} else {
		log.SetOutput(os.Stdout)
	}

	// Basic level filtering (you might want a more sophisticated approach for production)
	switch strings.ToLower(level) {
	case "debug":
		// No special filtering for now, everything logs
	case "info":
		// No special filtering for now, everything logs
	case "warn":
		// Can add logic to filter out info/debug logs
	case "error":
		// Can add logic to filter out info/debug/warn logs
	default:
		log.Printf("Unknown log level '%s', defaulting to info.", level)
	}
}
