#!/bin/bash
QUEUE="TripleThreat-Queue-V3"
REGION="us-west-2"
STATUSES=("SUBMITTED" "PENDING" "RUNNABLE" "STARTING" "RUNNING" "SUCCEEDED" "FAILED")

echo "----------------------------------------------------------------------"
echo " Checking All States for $QUEUE"
echo "----------------------------------------------------------------------"

for STATUS in "${STATUSES[@]}"; do
    echo -n "Status: $STATUS... "
    # Run the command and capture output
    RESULT=$(aws batch list-jobs --job-queue $QUEUE --region $REGION --job-status $STATUS --query "jobSummaryList[*].[jobId, status, statusReason]" --output table)
    
    # Check if the result contains actual data (more than just the table header)
    if [[ $(echo "$RESULT" | wc -l) -gt 3 ]]; then
        echo "FOUND"
        echo "$RESULT"
    else
        echo "Empty"
    fi
done
