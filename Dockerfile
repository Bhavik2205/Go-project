FROM golang:1.22.4-bullseye

# Install Python and pip
RUN apt update && apt install -y python3 python3-pip

WORKDIR /app
COPY . .

# Install Go dependencies
RUN go mod download

# Set environment for ONNX runtime
ENV LD_LIBRARY_PATH=/app/models/runtime/linux

CMD ["go", "run", "./cmd/main.go"]
