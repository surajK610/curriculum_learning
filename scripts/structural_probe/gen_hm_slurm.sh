#!/bin/sh

# Function to generate a SLURM script for a given directory
generate_slurm_script() {
  local dir=$1
  local name=$2
  local dirpull=$3
  local output_script="${dir}/hm_${name}_experiments.sh"
  local count=0
  
  # Start the SLURM script with the header
  echo "#!/bin/sh" > "$output_script"
  echo "#SBATCH --time=24:00:00" >> "$output_script"
  echo "#SBATCH --mem=64G" >> "$output_script"
  echo "#SBATCH -J hm_probe_seed_0_${dir}" >> "$output_script"
  echo "#SBATCH -o ./outputs/structural/probe/%x_%A_%a.out" >> "$output_script"
  echo "#SBATCH -e ./outputs/structural/probe/%x_%A_%a.err" >> "$output_script"
  # Uncomment the next line if the GPU partition is required
  echo "#SBATCH -p gpu --gres=gpu:1" >> "$output_script"
  echo "#SBATCH --array=0-$(expr $(ls -1 "${dir}"/*.yaml | wc -l) - 1)" >> "$output_script"
  echo "" >> "$output_script"
  
  # Write the command array
  for file in "${dirpull}"/*.yaml; do
    echo "if [ \"\$SLURM_ARRAY_TASK_ID\" -eq $count ]; then" >> "$output_script"
    echo "  python structural-probes/structural-probes/run_experiment.py $file" >> "$output_script"
    echo "fi" >> "$output_script"
    echo "" >> "$output_script"
    count=$((count+1))
  done
}

# Create output directories
mkdir -p ./job
mkdir -p ./job
mkdir -p ./outputs/structural/probe

# Generate the SLURM script for each directory
generate_slurm_script "./job" "depth_higher" "./data/structural-probes/yaml_variations/depth_higher"
# generate_slurm_script "./job" "distance" "./data/structural-probes/yaml_variations/distance"

echo "SLURM scripts generated."