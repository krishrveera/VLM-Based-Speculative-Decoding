#!/bin/bash
# Run on YOUR laptop (or in Cloud Shell: https://shell.cloud.google.com), AFTER the coupon is redeemed.
# Creates the project, links the coupon billing account, enables Compute, adds teammates, sets a budget.
#
#   bash 1_gcp_project_setup.sh
#
# Needs the gcloud CLI (https://cloud.google.com/sdk/docs/install) — Cloud Shell already has it.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-vlm-specdec-$(whoami | tr -cd 'a-z0-9' | head -c 8)}"   # must be globally unique
REGION="us-central1"
TEAMMATES=()            # optional, e.g. TEAMMATES=("uni1@columbia.edu" "uni2@columbia.edu")

gcloud auth login --update-adc            # opens a browser; sign in with the account that redeemed the coupon

echo ">> Billing accounts on this login (the coupon one usually says 'Billing Account for Education'):"
gcloud billing accounts list
read -rp "Paste the BILLING ACCOUNT ID of the coupon (XXXXXX-XXXXXX-XXXXXX): " BILLING_ID

echo ">> Creating project $PROJECT_ID"
gcloud projects create "$PROJECT_ID" --name="VLM SpecDec" || echo "(project may already exist — continuing)"
gcloud config set project "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ID"

echo ">> Enabling APIs"
gcloud services enable compute.googleapis.com storage.googleapis.com cloudquotas.googleapis.com
gcloud config set compute/region "$REGION"

echo ">> Budget alerts at 50% and 90% of \$100"
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create --billing-account="$BILLING_ID" --display-name="vlm-specdec-100" \
  --budget-amount=100USD --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
  || echo "(budget creation failed — set it in the console: Billing → Budgets & alerts)"

echo ">> Results bucket"
gsutil mb -l "$REGION" "gs://${PROJECT_ID}-results" || echo "(bucket may already exist)"

for m in "${TEAMMATES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="user:$m" --role="roles/compute.admin" >/dev/null
  echo "added $m"
done

echo ">> Current GPU quota in $REGION (LIMIT is what you can use; new projects show 0):"
gcloud compute regions describe "$REGION" --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  --flatten=quotas | grep -Ei "metric|GPU" || true
echo
echo "NEXT: request GPU quota in the console (gcloud can't file the request on all accounts):"
echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
echo "  1) 'GPUs (all regions)'            -> 1"
echo "  2) 'NVIDIA A100 GPUs' us-central1  -> 1   (also 'Preemptible NVIDIA A100 GPUs' -> 1 for spot)"
echo "  3) 'NVIDIA L4 GPUs'   us-central1  -> 1   (cheap setup box / fallback)"
echo "  Justification: 'Columbia COMS E6998 course project: benchmarking 7B vision-language model inference.'"
