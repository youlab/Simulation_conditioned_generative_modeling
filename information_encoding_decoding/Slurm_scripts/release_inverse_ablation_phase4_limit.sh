#!/bin/bash
# Run with bash, not sbatch. Release the existing array; do not submit duplicates.
set -e

# Zero removes the array's concurrency throttle. Cluster resource limits still apply.
scontrol update JobId=59187906 ArrayTaskThrottle=0
squeue -j 59187906 -o '%.25i %.40j %.9T %R'
