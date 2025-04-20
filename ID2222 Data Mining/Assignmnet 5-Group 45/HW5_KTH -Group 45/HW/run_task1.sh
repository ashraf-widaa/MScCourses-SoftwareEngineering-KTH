#!/bin/bash

# Variables which will go inside for-loops
graphs="3elt add20"
node_policies="LOCAL HYBRID"
# Default delta values for linear annealing policy
delta_linear="0.003"

# Check if 'Task1' folder exists and remove its contents
if [ -d "Task1" ]; then
  rm -r Task1/*
fi

# Check if 'Task1_results' folder exists and remove its contents or create it
if [ -d "Task1_results" ]; then
  rm -r Task1_results/*
else
  mkdir Task1_results
fi

# Check if 'results' folder exists and create it if not or remove its contents
if [ ! -d "results" ]; then
  mkdir results
else
  rm -r results/*
fi

for graph in $graphs; do
  # Task 1
  for node_policy in $node_policies; do
    # Run the script with appropriate parameters
    bash run.sh -outputDir Task1 -graph "./graphs/${graph}.graph" -delta "$delta_linear" -nodeSelectionPolicy "$node_policy"

    # Check if the 'results' folder exists
    if [ -d "results" ]; then
      # Move contents from 'results' to 'Task1_results'
      mv results/* Task1_results/  :
    else
      echo "Error: 'results' folder not found after Task1 execution."
      # Add any necessary error handling here
    fi

    # Run the Python script for plotting
    python plot.py Task1

    # Logging
    echo "Task1 completed for graph: ${graph}, node policy: ${node_policy}"
  done
done
