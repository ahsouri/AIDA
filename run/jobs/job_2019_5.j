#!/bin/bash 
#PBS -l select=4:ncpus=7:mpiprocs=7:model=has  
#PBS -l walltime=8:00:00  
#PBS -N aida  
#PBS -j oe  
#PBS -m abe 
#PBS -o aida_20195.out  
#PBS -e aida_20195.err  
#PBS -W group_list=s1395 
cd $PBS_O_WORKDIR  
source /nobackup/jjung13/miniconda3/bin/activate 
python ./job.py 2019 5