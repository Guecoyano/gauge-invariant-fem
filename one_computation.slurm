#!/bin/bash
#SBATCH --job-name=one_computation
#SBATCH --ntasks=12		# number of MPI processes
#SBATCH --cpus-per-task=2	# number of threads per MPI Process
#SBATCH -o outputlogs/mylog.%J.out		# standard output
#SBATCH -e outputlogs/mylog.%J.err		# standard error
#SBATCH --nodes=1

echo "Job started on `date`"
echo "-=-"
cat one_computation.slurm
module load python-anaconda3-2019

echo "execute data_one_computation.py"
cat mag_vec_fem/data_one_computation.py
python mag_vec_fem/data_one_computation.py $*

echo "-=-"
echo "Job ended on `date`"