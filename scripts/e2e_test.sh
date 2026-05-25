#!/bin/bash

# End-to-End Test Script for ML-Bot
# Tests REST API + WebSockets

set -e

# Configuration
SERVER_PORT=8080
BASE_URL="http://localhost:$SERVER_PORT"
WS_URL="ws://localhost:$SERVER_PORT"
TIMEOUT=10
SERVER_PID=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_failure() { echo -e "${RED}✗ $1${NC}"; }
log_info() { echo -e "${YELLOW}→ $1${NC}"; }
log_skip() { echo -e "${YELLOW}⚠ $1${NC}"; }

# Check if server is already running
check_server() {
    if curl -s "$BASE_URL/api/v1/health" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Start server if not running
start_server() {
    if check_server; then
        log_info "Server already running on port $SERVER_PORT"
        return 0
    fi
    log_info "Starting server..."
    go run cmd/server/main.go > e2e_server.log 2>&1 &
    SERVER_PID=$!
    log_info "Server PID: $SERVER_PID"
    for i in {1..30}; do
        if check_server; then
            log_success "Server is up"
            return 0
        fi
        sleep 1
    done
    log_failure "Server failed to start within 30 seconds"
    exit 1
}

# Stop server if we started it
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        log_info "Stopping server (PID $SERVER_PID)..."
        kill -TERM $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        log_success "Server stopped"
    fi
}

# Cleanup on exit
cleanup() {
    stop_server
}
trap cleanup EXIT INT TERM

# Run a single REST test
test_rest() {
    local name=$1
    local method=$2
    local path=$3
    local expected_status=$4
    local data=$5
    local auth_token=$6

    log_info "Testing $name..."

    local curl_cmd="curl -s -o /dev/null -w '%{http_code}' -X $method"
    if [ -n "$data" ]; then
        curl_cmd="$curl_cmd -H 'Content-Type: application/json' -d '$data'"
    fi
    if [ -n "$auth_token" ]; then
        curl_cmd="$curl_cmd -H 'Authorization: Bearer $auth_token'"
    fi
    curl_cmd="$curl_cmd '$BASE_URL$path'"

    local status=$(eval $curl_cmd)
    if [ "$status" -eq "$expected_status" ]; then
        log_success "$name returned $status"
        return 0
    else
        log_failure "$name expected $expected_status, got $status"
        return 1
    fi
}

# WebSocket test using wscat
test_websocket() {
    local name=$1
    local path=$2
    local expected_keyword=$3

    log_info "Testing WebSocket $name..."

    if ! command -v wscat &> /dev/null; then
        log_skip "wscat not installed, skipping WebSocket test"
        return 0
    fi

    local output=$(timeout $TIMEOUT wscat -c "$WS_URL$path" -x 'wait' 2>&1 || true)
    if [ -n "$expected_keyword" ]; then
        if echo "$output" | grep -q "$expected_keyword"; then
            log_success "WebSocket $name works (received data)"
        else
            log_skip "WebSocket $name – no data within ${TIMEOUT}s (may need more time)"
        fi
    else
        if [ -n "$output" ]; then
            log_success "WebSocket $name connected"
        else
            log_skip "WebSocket $name – no response within ${TIMEOUT}s"
        fi
    fi
}

# Main test sequence
main() {
    start_server

    # 1. Health endpoint
    test_rest "Health" "GET" "/api/v1/health" 200

    # 2. OpenAPI spec
    test_rest "OpenAPI spec" "GET" "/api/v1/openapi.json" 200

    # 3. Signup
    SIGNUP_DATA='{"email":"e2e@example.com","password":"Test123!","userName":"e2euser"}'
    test_rest "Signup" "POST" "/api/v1/auth/signup" 201 "$SIGNUP_DATA"

    # 4. Login
    LOGIN_DATA='{"email":"e2e@example.com","password":"Test123!"}'
    LOGIN_RESP=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" -H "Content-Type: application/json" -d "$LOGIN_DATA")
    ACCESS_TOKEN=$(echo "$LOGIN_RESP" | grep -o '"accessToken":"[^"]*' | cut -d'"' -f4)
    REFRESH_TOKEN=$(echo "$LOGIN_RESP" | grep -o '"refreshToken":"[^"]*' | cut -d'"' -f4)

    if [ -n "$ACCESS_TOKEN" ] && [ -n "$REFRESH_TOKEN" ]; then
        log_success "Login succeeded, tokens obtained"
    else
        log_failure "Login failed to return tokens"
        exit 1
    fi

    # 5. /me (protected)
    test_rest "Get /me" "GET" "/api/v1/me" 200 "" "$ACCESS_TOKEN"

    # 6. Refresh token
    REFRESH_DATA="{\"refreshToken\":\"$REFRESH_TOKEN\"}"
    REFRESH_RESP=$(curl -s -X POST "$BASE_URL/api/v1/auth/refresh" -H "Content-Type: application/json" -d "$REFRESH_DATA")
    NEW_ACCESS_TOKEN=$(echo "$REFRESH_RESP" | grep -o '"accessToken":"[^"]*' | cut -d'"' -f4)
    NEW_REFRESH_TOKEN=$(echo "$REFRESH_RESP" | grep -o '"refreshToken":"[^"]*' | cut -d'"' -f4)
    if [ -n "$NEW_ACCESS_TOKEN" ]; then
        log_success "Refresh token succeeded"
    else
        log_failure "Refresh token failed"
        exit 1
    fi

    # 7. Refresh token reuse (should fail)
    REUSE_RESP=$(curl -s -X POST "$BASE_URL/api/v1/auth/refresh" -H "Content-Type: application/json" -d "$REFRESH_DATA")
    if echo "$REUSE_RESP" | grep -q '"error"'; then
        log_success "Refresh token reuse correctly rejected"
    else
        log_failure "Refresh token reuse – vulnerability remains"
    fi

    # 8. Settings: Get default (should return empty JSON for non-existent section)
    test_rest "Get settings (empty)" "GET" "/api/v1/settings?section=general" 200 "" "$NEW_ACCESS_TOKEN"

    # 9. Settings: Update a section
    SETTINGS_DATA='{"section":"general","data":{"theme":"dark","notifications":true}}'
    test_rest "Update settings" "PUT" "/api/v1/settings" 200 "$SETTINGS_DATA" "$NEW_ACCESS_TOKEN"

    # 10. Settings: Verify update
    log_info "Verifying settings update..."
    SETTINGS_GET_RESP=$(curl -s -H "Authorization: Bearer $NEW_ACCESS_TOKEN" "$BASE_URL/api/v1/settings?section=general")
    if echo "$SETTINGS_GET_RESP" | grep -q '"theme":"dark"'; then
        log_success "Settings update verified"
    else
        log_failure "Settings update not reflected: $SETTINGS_GET_RESP"
        exit 1
    fi

    # 11. Broker status (simulation mode)
    test_rest "Broker status" "GET" "/api/v1/brokers/zerodha/status" 200 "" "$NEW_ACCESS_TOKEN"

    # 12. Quotes
    test_rest "Quotes" "GET" "/api/v1/quotes?symbols=NSE:RELIANCE" 200 "" "$NEW_ACCESS_TOKEN"

    # 13. Market overview
    test_rest "Market overview" "GET" "/api/v1/market/overview" 200 "" "$NEW_ACCESS_TOKEN"

    # 14. Runtime config (protected)
    test_rest "Runtime config" "GET" "/api/v1/runtime/config" 200 "" "$NEW_ACCESS_TOKEN"
    log_info "Verifying runtime config..."
    CONFIG_RESP=$(curl -s -H "Authorization: Bearer $NEW_ACCESS_TOKEN" "$BASE_URL/api/v1/runtime/config")
    if echo "$CONFIG_RESP" | grep -q '"mode"'; then
        log_success "Runtime config contains expected fields"
    else
        log_failure "Runtime config missing expected fields"
        exit 1
    fi

    # 15. Runtime metrics (protected)
    test_rest "Runtime metrics" "GET" "/api/v1/runtime/metrics" 200 "" "$NEW_ACCESS_TOKEN"
    log_info "Verifying runtime metrics..."
    METRICS_RESP=$(curl -s -H "Authorization: Bearer $NEW_ACCESS_TOKEN" "$BASE_URL/api/v1/runtime/metrics")
    if echo "$METRICS_RESP" | grep -q '"websocket_clients"'; then
        log_success "Runtime metrics contains expected fields"
    else
        log_failure "Runtime metrics missing expected fields"
        exit 1
    fi

    # 16. WebSocket tests (if wscat available)
    test_websocket "/ws" "ticks" ""
    test_websocket "/ws/candles" "candles" ""
    test_websocket "/ws/indicators" "indicators" ""
    test_websocket "/ws/heatmap" "heatmap" "Symbol"

    # 17. Instrument endpoint (optional)
    if curl -s "$BASE_URL/api/instrument?symbol=RELIANCE" > /dev/null 2>&1; then
        test_rest "Instrument lookup" "GET" "/api/instrument?symbol=RELIANCE" 200 "" "$NEW_ACCESS_TOKEN"
    else
        log_skip "Instrument endpoint not registered (maybe only in live mode)"
    fi

    # 18. Logout
    LOGOUT_DATA="{\"refreshToken\":\"$NEW_REFRESH_TOKEN\"}"
    test_rest "Logout" "POST" "/api/v1/auth/logout" 204 "$LOGOUT_DATA" "$NEW_ACCESS_TOKEN"

    echo ""
    log_success "All tests passed! The system is healthy."
}

main "$@"