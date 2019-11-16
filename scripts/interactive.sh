# to see all gpus any pci express
cat /sailhome/slurm/gres.conf

srun -p dgx --pty bash

# 4x each set to 75%
srun -p napoli-gpu --nodelist=napoli101,napoli102,napoli103,napoli104 --pty bash

# 1080 ti
srun -p napoli-gpu --nodelist=napoli111,napoli112 --pty bash

# 30x 50x faster core (volta arch, anything with v)
# 2x fastest card is titan v100 on napoli108
srun -p napoli-gpu --gres=gpu:p100:2 --pty bash

sbatch -p napoli-gpu --nodelist=napoli101 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli102 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli103 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli104 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli105 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli106 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli107 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli108 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli109 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli110 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli111 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh
sbatch -p napoli-gpu --nodelist=napoli112 --time=06:00:00 --nodes=1 --ntasks-per-node=1 --mem=2G ./scripts/scr_setup.sh

srun -p napoli-gpu --gres=gpu:1080ti:4 --mem=42G  --cpus-per-task=8 --pty bash
