#!/bin/bash

# Variables which will go inside for-loops
graphs="3elt add20 Twitter" # Add other graphs separated by space
# Best delta values found per annealing policy
delta_linear="0.003"
d_exp="0.9"

# Check if 'Task2.2_results' folder exists and remove its contents
if [ -d "Task2.2_results" ]; then
  rm -r Task2.2_results/*
else
  mkdir Task2.2_results
fi

# Check if 'results' folder exists and create it if not
if [ ! -d "results" ]; then
  mkdir results
else
  rm -r results/*
fi

# Task 2.2 - Explore Restart Options
for graph in $graphs; do
  # Linear Annealing
  echo "Running linear annealing for graph: $graph"
  bash run.sh -outputDir Task2.2 -graph "./graphs/${graph}.graph" -delta $delta_linear -optionAnnealing linear -roundtoRestart 100 -temp 2
  # Exponential Annealing
  echo "Running exponential annealing for graph: $graph"
  bash run.sh -outputDir Task2.2 -graph "./graphs/${graph}.graph" -delta $d_exp -optionAnnealing exp -roundtoRestart 100 -temp 2
done

# Plot the results using Python script
python plot.py Task2.2

# Move contents from 'results' to 'Task2.2_results'
mv results/* Task2.2_results/

# Check if the 'results' folder exists
if [ -d "results" ]; then
  # Remove 'results' folder after moving its contents
  rm -r results/*
else
  echo "Error: 'results' folder not found after Task2 execution."
  # Add any necessary error handling here
fi

# Logging
echo "Task2 completed for all graphs"
