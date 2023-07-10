#!/bin/bash
#SBATCH --job-name=one_computation
#SBATCH --ntasks=12		# number of MPI processes
#SBATCH --cpus-per-task=2	# number of threads per MPI Process
#SBATCH -o outputlogs/mylog.%J.out		# standard output
#SBATCH -e outputlogs/mylog.%J.err		# standard error
#SBATCH --nodes=1


today=$( date +%Y%m%d )   # or: printf -v today '%(%Y%m%d)T' -1
number=0

fname="${today}"
DIR0="mag_vec_fem/data/${fname}"
DIR=$DIR0
while [ -e "$DIR" ]; do
    printf -v DIR '%s-%02d' "$DIR0" "$(( ++number ))"
done
mkdir "$DIR"
DIR_TO_SAVE=${DIR}/plots
NCYCLES=12
NBATCH=10
ETA="2e-1"
BETA="2e-1"
NAME_PR="PR_eta${ETA}beta${BETA}"

echo "Job started on `date`"
echo "-=-"
cat IPR_serialsolve.slurm

module load python-anaconda3-2019
python mag_vec_fem/IPR_script.py N_eig=$NBATCH serial_solve=False
for foo in $(seq 2 $NCYCLES)
do
    python mag_vec_fem/IPR_script.py N_eig=$NBATCH serial_solve=True h=1e-3 eta=$ETA beta=$BETA dir_to_save=$DIR_TO_SAVE name_pr=$NAME_PR
