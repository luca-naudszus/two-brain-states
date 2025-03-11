#!/bin/bash

input_file=$1 
output_file=$2

# Extract max number of clusters from the file
max_clusters=$(grep -oP 'n_clusters \K\d+' "$input_file" | sort -nr | head -1)

# Generate CSV header dynamically
header="Iteration,Shrinkage,Kernel,N_clusters,Silhouette Score,Calinski-Harabasz Score"
for ((i=1; i<=max_clusters; i++)); do
    header+=",Riemannian Variance $i"
done
header+=",Davies-Bouldin-Index,Geodesic Distance Ratio"
echo "$header" > "$output_file"

awk -v out_file="$output_file" -v max_clusters="$max_clusters" '
BEGIN {
    iteration=""; shrinkage=""; kernel=""; n_clusters="";
    silhouette=""; ch_score=""; dbi=""; gdr="";
    split("", riemannian_variance); # Clear array
}
!/FutureWarning/ && !/Traceback/ && !/ValueError/ && !/Elapsed time/ {
    if ($0 ~ /^Iteration/) {
        if (iteration != "") {
            # Prepare the Riemannian Variance fields, ensuring the correct number of columns
            rv_string = ""
            for (i = 1; i <= max_clusters; i++) {
                if (i <= length(riemannian_variance)) {
                    rv_string = rv_string "," riemannian_variance[i]
                } else {
                    rv_string = rv_string ","
                }
            }
            print iteration "," shrinkage "," kernel "," n_clusters "," silhouette "," ch_score rv_string "," dbi "," gdr >> out_file
        }
        match($0, /Iteration ([0-9]+), parameters: shrinkage ([0-9.]+), kernel ([^,]+), n_clusters ([0-9]+)/, arr)
        iteration = arr[1]; shrinkage = arr[2]; kernel = arr[3]; n_clusters = arr[4]
        silhouette=""; ch_score=""; dbi=""; gdr="";
        split("", riemannian_variance); # Reset the array
    }
    if ($0 ~ /^Silhouette Score:/) {
        match($0, /Silhouette Score: ([-0-9.nan]+)/, arr)
        silhouette = arr[1]
    }
    if ($0 ~ /^Calinski-Harabasz Score:/) {
        match($0, /Calinski-Harabasz Score: ([0-9.eE+-]+)/, arr)
        ch_score = arr[1]
    }
    if ($0 ~ /^Riemannian Variance:/) {
        gsub(/.*\[/, "", $0); gsub(/\].*/, "", $0)
        gsub(/np.float64\(/, "", $0); gsub(/\)/, "", $0)
        split($0, riemannian_variance, ", ")
    }
    if ($0 ~ /^Davies-Bouldin-Index:/) {
        match($0, /Davies-Bouldin-Index: ([0-9.eE+-]+)/, arr)
        dbi = arr[1]
    }
    if ($0 ~ /^Geodesic Distance Ratio:/) {
        match($0, /Geodesic Distance Ratio: ([0-9.eE+-]+)/, arr)
        gdr = arr[1]
    }
}
END {
    if (iteration != "") {
        rv_string = ""
        for (i = 1; i <= max_clusters; i++) {
            if (i <= length(riemannian_variance)) {
                rv_string = rv_string "," riemannian_variance[i]
            } else {
                rv_string = rv_string ","
            }
        }
        print iteration "," shrinkage "," kernel "," n_clusters "," silhouette "," ch_score rv_string "," dbi "," gdr >> out_file
    }
}' "$input_file"

echo "CSV file generated: $output_file"
