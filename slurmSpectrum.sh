#!/bin/bash -x
if [ "x$SLURM_NPROCS" = "x" ]
then
	if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
	then
		SLURM_NTASKS_PER_NODE=1
	fi
	SLURM_NPROCS='expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE'
else
	if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
	then
		SLURM_NTASKS_PER_NODE='expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES'
	fi
fi
srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID >/tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

module load gcc/7.4.0/1
module load spectrum-mpi
module load cuda

mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS covid-cuda-mpi-exe 50 128 128 2 0.25 20 1
rm /tmp/hosts.$SLURM_JOB_ID
