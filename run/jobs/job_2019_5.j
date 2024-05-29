#!/bin/bash 
#PBS -l select=2:ncpus=24:mpiprocs=24:model=has  
#PBS -l walltime=2:00:00  
#PBS -N aida  
#PBS -j oe  
#PBS -m abe 
#PBS -M helloiamjia@gmail.com 
#PBS -o aida_20195.out  
#PBS -e aida_20195.err  
#PBS -W group_list=s1395 
#PBS -q devel  
cd $PBS_O_WORKDIR  
/nobackup/jjung13/miniconda3/bin/python3 ./job.py 2019 5