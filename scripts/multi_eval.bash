#!/bin/bash

# Define the base command with placeholders
BASE_COMMAND="python test.py dataset=munich use_cloud=true use_image=false use_footprint=true wandb=false run_suffix='_alto' gpu_id=3 test.check_point='./outputs/\${test.run_name}\${run_suffix}/check_points/model_"

# Loop over checkpoint indices from 100 to 10000 in increments of 100
for i in {100..10000..100}
do
    # Construct the full command for each checkpoint
    FULL_COMMAND="${BASE_COMMAND}${i}.pt'"

    # Execute the command
    eval $FULL_COMMAND
done
