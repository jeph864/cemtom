#!/bin/bash
#SBATCH -J topics_nb                      # the job name
#SBATCH --nodes=1             # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16             # use 1 thread per taks
#SBATCH -N 1
#SBATCH --partition=informatik-mind
#SBATCH --output=output/jupyter_bert_out.txt         # capture output
#SBATCH --error=output/jupyter_bert_err.txt          # and error streams

module purge
module add nvidia
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
echo "activating the cemtom env"
conda activate cemtom
conda info

pip list
PROJECT="/scratch/$USER/thesis/tms/atlas"
cd $PROJECT
export TOKENIZERS_PARALLELISM=true

#list loaded modules
module list
nvidia-smi
#start notebook
node=$(hostname -s)
#start jupyter
jupyter-notebook --no-browser  --ip=${node}
#python bert.py
