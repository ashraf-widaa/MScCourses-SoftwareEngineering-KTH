import os
import pandas as pd
import matplotlib.pyplot as plt

# Define folder names
input_folder = "Task1_test"  # Change this to the actual input folder name
output_folder = "output_images"  # Change this to the actual output folder name

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List to store results
results = []

# Loop through each text file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)

        # Read the content of the text file
        with open(filepath, "r") as file:
            content = file.readlines()

        # Check if the content has enough lines
        if len(content) < 8:
            print(f"Skipping {filename} due to insufficient content.")
            continue

        # Parse the content and extract relevant information
        graph = content[0].strip()
        annealing_policy = content[1].strip()
        node_policy = content[2].strip()

        try:
            # Replace tabs with spaces and split the string
            values = content[3].replace('\t', ' ').split()

            # Check if the length of values is sufficient
            if len(values) < 5:
                raise ValueError(f"Insufficient values in {filename}. Expected 5, got {len(values)}.")

            delta = float(values[0])
            delta_decay = float(values[1])
            edge_cut = int(values[2])
            swaps = int(values[3])
            migrations = int(values[4])
        except (ValueError, IndexError) as e:
            print(f"Error parsing values in {filename}: {e}")
            continue

        # Append the results to the list
        results.append({
            'Graph': graph,
            'Annealing': annealing_policy,
            'Node Policy': node_policy,
            'Delta': delta,
            'Delta Decay': delta_decay,
            'Edge-cut': edge_cut,
            'Swaps': swaps,
            'Migrations': migrations
        })

        # Create a DataFrame from the results list
        results_df = pd.DataFrame(results)

        # Remove columns which contain only one possible value
        for column in ['Annealing', 'Node Policy', 'Delta', 'Delta Decay']:
            if column in results_df.columns and len(results_df[column].unique()) == 1:
                results_df.drop(column, axis=1, inplace=True)

        # Display the resulting DataFrame
        print(results_df)

        # Save the DataFrame as a PNG file
        base_name = os.path.splitext(filename)[0]
        png_filepath = os.path.join(output_folder, f'{base_name}_table.png')
        plt.axis('off')
        plt.table(cellText=results_df.values,
                  colLabels=results_df.columns,
                  cellLoc='center', loc='center')
        plt.savefig(png_filepath, bbox_inches='tight')
        plt.close()

        print(f"Processing {filename} complete.")

        # Move the PNG file to the specified folder
        os.rename(png_filepath, os.path.join(output_folder, f'{base_name}_table.png'))

print("All files processed.")
