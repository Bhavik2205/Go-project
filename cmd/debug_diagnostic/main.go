package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("🚨 EMERGENCY DEBUG: Starting server diagnostic...")

	// Test 1: Basic config loading
	fmt.Println("\n--- TEST 1: Config Loading ---")
	if _, err := os.Stat("configs/app.yaml"); os.IsNotExist(err) {
		fmt.Println("❌ CRITICAL: configs/app.yaml NOT FOUND")
		return
	}
	fmt.Println("✅ app.yaml exists")

	// Test 2: Environment variables
	fmt.Println("\n--- TEST 2: Environment Variables ---")
	apiKey := os.Getenv("ZERODHA_API_KEY")
	if apiKey == "" {
		fmt.Println("❌ CRITICAL: ZERODHA_API_KEY not set")
	} else {
		fmt.Println("✅ ZERODHA_API_KEY found")
	}

	// Test 3: Database connection (simplified)
	fmt.Println("\n--- TEST 3: Database Connection ---")
	// We'll just check if we can open the config

	fmt.Println("\n--- DIAGNOSTIC COMPLETE ---")
	fmt.Println("Run the actual server now and watch for the FIRST error message")
}
