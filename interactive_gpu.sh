#!/bin/bash
# ==============================================================================
# Helper Script to Launch an Interactive GPU Session on Ada (IIIT-H)
# Follows Ada 1:10 GPU-to-CPU ratio policy.
# ==============================================================================

ACCOUNT="${1:-research}" # Default account is research. Change to cvit/nlp if applicable.
PARTITION="${2:-long}"   # 'long' for nodes 03-92, 'short' for nodes 01-02 (max 6h)

echo "Requesting interactive session on Ada..."
echo "Account:   $ACCOUNT"
echo "Partition: $PARTITION"
echo "Resources: 1 GPU, 10 CPUs, 20GB RAM (2G/CPU)"
echo "----------------------------------------------------------------------"

# srun invocation with proper flags
srun --pty \
    --partition="$PARTITION" \
    -A "$ACCOUNT" \
    --gres=gpu:1 \
    --mem-per-cpu=2G \
    -c 10 \
    bash -l
