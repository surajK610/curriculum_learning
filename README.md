# transformer-curvature

If running this code in the Oscar (SLURM), first run `scripts/env_setup.sh` to generate the virtual environment

## Dataset Creation/Experiment Running

Assumes slurm.

To run experiments, please specify parameters and make changes in the `*.json` files in `datasets/` and `experiments/`.

```bash
# Sets up folders.
setup.sh 

# Create dataset.
sbatch configs/datasets/pipeline.sh
sbatch configs/experiments/pipeline.sh
```