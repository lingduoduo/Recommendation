#!/usr/bin/env bash

PORT=8080
echo "Port: $PORT"

# POST method predict
curl -d '{
   "user": "tagged:6145378585",
   "path": "train_gift_two_tower"
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict
