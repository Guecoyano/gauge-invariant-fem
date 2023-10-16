#!/bin/bash
#SBATCH --job-name=one_computation
#SBATCH --ntasks=12		# number of MPI processes
#SBATCH --cpus-per-task=2	# number of threads per MPI Process
#SBATCH -o outputlogs/mylog.%J.out		# standard output
#SBATCH -e outputlogs/mylog.%J.err		# standard error
#SBATCH --nodes=1

echo "Job started on `date`"
echo "-=-"
cat serial_computation.slurm
module load python-anaconda3-2019

today=$( date +%Y%m%d )   # or: printf -v today '%(%Y%m%d)T' -1
number=0

fname="${today}"
DIR0="mag_vec_fem/data/${fname}"
DIR=$DIR0
while [ -e "$DIR" ]; do
    printf -v DIR '%s-%02d' "$DIR0" "$(( ++number ))"
done
mkdir "$DIR"
DIR_TO_SAVE=${DIR}
NBATCH=10
ETA="2e-1"
BETA="2e-1"
NCYCLES=12
#if one passes arguments to the script such as eta, beta, v, and arguments to be passed to the python script, one may use 
#ETA=$1; BETA=$2; NCYCLES=$3; V=$4; shift; shift; shift; shift

H="1e-3"
V="Na400x15sig22v0"
NAME_EIG="${V}Symh1000eta${ETA}beta${BETA}"

echo "DIR=${DIR} NBATCH=$NBATCH ETA=$ETA BETA=$BETA NCYCLES=$NCYCLES H=$H V=$V NAME_EIG=$NAME_EIG"

python mag_vec_fem/data_one_computation.py h=$H N_eig=$NBATCH serial_solve=False dir_to_save=$DIR_TO_SAVE name_eig="${NAME_EIG}part1" eta=$ETA beta=$BETA
for foo in $(seq 2 $NCYCLES)
do
    NAME_EIG="${V}Symh1000eta${ETA}beta${BETA}"
    python mag_vec_fem/data_one_computation.py h=$H N_eig=$NBATCH serial_solve=True eta=$ETA beta=$BETA dir_to_save=$DIR_TO_SAVE name_eig=$NAME_EIG part=$foo
done

echo "-=-"
echo "Job ended on `date`"

