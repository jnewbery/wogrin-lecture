#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.env"

IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "Building and pushing image: $IMAGE"
gcloud builds submit --tag "$IMAGE"

echo "Deploying to Cloud Run (region: $REGION)"
gcloud run deploy $SERVICE_NAME \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated

echo "Done. Service URL:"
gcloud run services describe $SERVICE_NAME --region "$REGION" --format "value(status.url)"
