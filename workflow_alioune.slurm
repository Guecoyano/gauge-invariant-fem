#!/bin/bash
#SBATCH --job-name=fe
#SBATCH --ntasks=12		# number of MPI processes
#SBATCH --cpus-per-task=1	# number of threads per MPI Process
#SBATCH -o outputlogs/mylog.%J.out		# standard output
#SBATCH -e outputlogs/mylog.%J.err		# standard error
#SBATCH --nodes=1

echo "Job started on `date`"
echo "-=-"
cat workflow.slurm
module purge

python store_xy_2d.py _50
mpirun -np 10 -hosts $SLURM_JOB_NODELIST python calc_eigen_couplings_hybrid_parallel.py _50
python find_fundamentals.py _50


# python ma_2d_all_purpose.py _50 400



#mpirun -np 12 -hosts $SLURM_JOB_NODELIST python calc_agmon_couplings_parallel.py
#mpiexec -n 8 -hosts $SLURM_JOB_NODELIST python eigen_solver_0.py -eps_view
#mpiexec -n 1 -hosts $SLURM_JOB_NODELIST python parallel.py
#python eigen_solver_0.py
#mpirun  -n 1 -hosts $SLURM_JOB_NODELIST python pbc_fenics_eigensolver.py
#python store_xy_mod.py

echo "-=-"
echo "Job ended on `date`"