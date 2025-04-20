#!/bin/bash
# Variables which will go inside for-loops
graphs="3elt add20 Twitter" # Add other graphs separated by space
# Best delta values found per annealing policy
delta_exp="0.9 0.8 0.6"

# Check if 'Task1_results_test' folder exists and remove its contents
if [ -d "Task2.1_results" ]; then
  rm -r Task2.1_results/*
else
  mkdir Task2.1_results
fi
# Check if 'results' folder exists and create it if not
if [ ! -d "results" ]; then
  mkdir results
  else
    rm -r results/*
fi

# Task 2.1 - Explore Exponential Annealing Options
for graph in $graphs; do
  for delta in $delta_exp; do
    bash run.sh -outputDir Task2.1 -graph ./graphs/${graph}.graph -delta $delta -optionAnnealing exp
  done
done

# Plot the results using Python script
python plot.py Task2.1
 # Move contents from 'results' to 'Task2_results'
mv results/* Task2.1_results/

# Check if the 'results' folder exists
if [ -d "results" ]; then
  # Remove 'results' folder after moving its contents
  rm -r results/*
else
  echo "Error: 'results' folder not found after Task2 execution."
  # Add any necessary error handling here
fi

# Logging
echo "Task2.1 completed for all graphs"
