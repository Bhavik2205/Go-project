#!/bin/bash

# db_migrate.sh
# This script manages database migrations using golang-migrate/migrate.

# --- Configuration ---
# Load environment variables from .env file (if it exists)
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Database connection details from environment variables or defaults
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-admin} # IMPORTANT: Use a strong password in .env for production!
DB_NAME=${DB_NAME:-trading_bot_db}   # Make sure this matches your configs/database.yaml

# Construct the database URL for migrate tool
# Note: Using 'disable' for sslmode for local development. Use 'require' or 'verify-full' for production.
DATABASE_URL="postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=disable"

# Path to your migration files
MIGRATION_PATH="internal/db/migrations"

# --- Functions ---

# Function to display usage
usage() {
    echo "Usage: $0 [up|down|create|force|version]"
    echo "  up [N]     : Apply N up migrations (or all if N is omitted)"
    echo "  down [N]   : Apply N down migrations (or all if N is omitted)"
    echo "  create <name> : Create a new migration file with the given name"
    echo "  force <version> : Force set the database version (use with caution!)"
    echo "  version    : Show the current database version"
    exit 1
}

# --- Main Logic ---
echo "DEBUG: DB_HOST=$DB_HOST"
echo "DEBUG: DB_PORT=$DB_PORT"
echo "DEBUG: DB_USER=$DB_USER"
echo "DEBUG: DB_PASSWORD=$DB_PASSWORD" # Be careful not to log this in production
echo "DEBUG: DB_NAME=$DB_NAME"
echo "DEBUG: DATABASE_URL=$DATABASE_URL"
echo "DEBUG: MIGRATION_PATH=$MIGRATION_PATH"

# Check if migrate command is available
if ! command -v migrate &> /dev/null
then
    echo "Error: 'migrate' command not found."
    echo "Please install it: go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest"
    exit 1
fi

COMMAND=$1
AMOUNT=$2

case "$COMMAND" in
    up)
        if [ -z "$AMOUNT" ]; then
            echo "Applying all up migrations..."
            migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" up
        else
            echo "Applying $AMOUNT up migrations..."
            migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" up "$AMOUNT"
        fi
        ;;
    down)
        if [ -z "$AMOUNT" ]; then
            echo "Applying all down migrations..."
            migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" down
        else
            echo "Applying $AMOUNT down migrations..."
            migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" down "$AMOUNT"
        fi
        ;;
    create)
        if [ -z "$AMOUNT" ]; then
            echo "Error: Migration name is required for 'create' command."
            usage
        fi
        echo "Creating new migration: $AMOUNT"
        migrate create -ext sql -dir "$MIGRATION_PATH" "$AMOUNT"
        ;;
    force)
        if [ -z "$AMOUNT" ]; then
            echo "Error: Version is required for 'force' command."
            usage
        fi
        echo "Forcing database version to: $AMOUNT"
        migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" force "$AMOUNT"
        ;;
    version)
        echo "Current database version:"
        migrate -path "$MIGRATION_PATH" -database "$DATABASE_URL" version
        ;;
    *)
        usage
        ;;
esac

if [ $? -eq 0 ]; then
    echo "Migration command executed successfully."
else
    echo "Migration command failed."
    exit 1
fi