import os
import sys

def generate_gnuplot_script(output_folder, folder_path, filename):
    script_content = f"""
set term png small size 1024,786
set output '{output_folder}/{filename}_graph.png'

set style line 1 lc rgb "#1a9850" lw 1.5
set style line 2 lc rgb "black" lw 1.5
set style line 3 lc rgb "brown" lw 1.5
set style line 4 lc rgb "green" lw 1.5
set style line 5 lc rgb "orange" lw 1.5
set style line 6 lc rgb "#d73027" lw 1.5

set multiplot layout 3,1 title "{filename}" font ",14"
set yrange [0:]

set ylabel "Edge Cut" 
set xlabel "Rounds" 
plot "{folder_path}/{filename}" using 1:2 with l ls 1 title "Edge-Cut"
set ylabel "Swaps" 
set xlabel "Rounds" 
plot "{folder_path}/{filename}" using 1:3 with l ls 2 title "Swaps"
set ylabel "Migrations" 
set xlabel "Rounds" 
plot "{folder_path}/{filename}" using 1:4 with l ls 3 title "Migrations"

unset multiplot
"""

    with open(f"{output_folder}/plot.gnuplot", "w") as gnuplot_script:
        gnuplot_script.write(script_content)

def run_gnuplot(output_folder):
    os.system(f"gnuplot {output_folder}/plot.gnuplot")

def plot_for_all_files(directory=".", output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            generate_gnuplot_script(output_folder, directory, filename)
            run_gnuplot(output_folder)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plotter.py <subdirectory>")
        sys.exit(1)

    plot_for_all_files(sys.argv[1])
