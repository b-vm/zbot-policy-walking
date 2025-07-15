#!/bin/bash

# Start Xvfb
Xvfb :101 -ac &

# Configure git safe directory
git config --global --add safe.directory /app

# Mount S3 if credentials are available
if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ] && [ ! -z "$S3_BUCKET" ]; then
    echo "Mounting S3 bucket for experiment data..."
    echo "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" > /root/.passwd-s3fs
    chmod 600 /root/.passwd-s3fs
    
    # Ensure local directory exists
    mkdir -p logs # local dir to write logs directly
    mkdir -p /mnt/logs # s3 bucket mount, to read previous experiment run numbers
    
    # Mount the S3 bucket
    s3fs "$S3_BUCKET:/zbot-policy-walking/logs" /mnt/logs \
        -o passwd_file=/root/.passwd-s3fs \
        -o umask=0077,uid=1000 \
        -o allow_other \
        -o readwrite_timeout=60 \
        -o retries=5 \
        -o dbglevel=info
    echo "S3 bucket mounted successfully at /mnt/logs"

    # Find next available run directory
    run_id=0
    while [ -d "/mnt/logs/run_$run_id" ]; do
        run_id=$((run_id + 1))
    done
    exp_dir="logs/run_$run_id"
    echo "Using experiment directory: $exp_dir"


    # Start periodic s3 sync in background. s3fs does not work well for tensorboard files.
    (while true; do
        sleep 600
        aws s3 sync "$exp_dir" "s3://$S3_BUCKET/zbot-policy-walking/$exp_dir" \
            --delete \
            --only-show-errors
        echo "Synced data to S3 at $(date)"

        # unmount s3 bucket after getting run id
        if mountpoint -q /mnt/logs; then
            fusermount -u /mnt/logs
            echo "Unmounted S3 bucket"
        fi
    done) &

    exec conda run -n ksim "$@" exp_dir=$exp_dir
else
    echo "AWS credentials not found - using local directory only"
    exec conda run -n ksim "$@"
fi
