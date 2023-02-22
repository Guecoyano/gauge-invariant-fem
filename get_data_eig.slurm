#!/bin/bash
#SBATCH --job-name=fe
#SBATCH --ntasks=12		# number of MPI processes
#SBATCH --cpus-per-task=1	# number of threads per MPI Process
#SBATCH -o outputlogs/mylog.%J.out		# standard output
#SBATCH -e outputlogs/mylog.%J.err		# standard error
#SBATCH --nodes=1

echo "Job started on `date`"
echo "-=-"
cat workflow_alioune.slurm
module purge
module load python-anaconda3-2019
module load texlive
pip install latex --user

echo "getting eigen values and functions and store in files"
cat mag_vec_fem/get_data_eig.py
python mag_vec_fem/get_data_eig.py


echo "-=-"
echo "Job ended on `date`"