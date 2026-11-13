#!/bin/bash
# Create the GPU VM once quota is approved. Run on your laptop / Cloud Shell.
#
#   bash 2_create_vm.sh l4      # cheap box for setup + sanity checks (~$0.3-0.4/h spot)
#   bash 2_create_vm.sh a100    # timed MMSpec runs (same GPU as the paper, ~$1.2-1.5/h spot)
#
# Spot VMs can be stopped by Google at any time; MMSpec skips finished samples, so just restart.
set -euo pipefail
KIND="${1:-l4}"
PROJECT_ID="$(gcloud config get-value project)"
case "$KIND" in
  a100) MACHINE=a2-highgpu-1g;  ZONES=(us-central1-a us-central1-b us-central1-c us-central1-f) ;;
  l4)   MACHINE=g2-standard-8;  ZONES=(us-central1-a us-central1-b us-central1-c) ;;
  *) echo "usage: $0 l4|a100"; exit 1 ;;
esac
NAME="mmspec-$KIND"

# Deep Learning VM image (PyTorch + CUDA). If this family is retired, list current ones with:
#   gcloud compute images list --project deeplearning-platform-release --filter="family~pytorch" --format="value(family)" | sort -u
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"

for Z in "${ZONES[@]}"; do
  echo ">> trying $MACHINE spot in $Z"
  if gcloud compute instances create "$NAME" --project="$PROJECT_ID" --zone="$Z" \
      --machine-type="$MACHINE" \
      --provisioning-model=SPOT --instance-termination-action=STOP \
      --maintenance-policy=TERMINATE \
      --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release \
      --boot-disk-size=200GB --boot-disk-type=pd-balanced \
      --metadata=install-nvidia-driver=True \
      --scopes=cloud-platform; then
    echo "$Z" > ".zone-$NAME"
    echo ">> created $NAME in $Z"
    echo "SSH:   gcloud compute ssh $NAME --zone $Z"
    echo "STOP:  gcloud compute instances stop $NAME --zone $Z     # do this whenever idle"
    exit 0
  fi
done
echo "No capacity in any zone right now. Try again later, or: bash $0 l4"
exit 1
