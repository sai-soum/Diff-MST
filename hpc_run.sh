#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -j y
#$ -pe smp 8      # Request 8 core
#$ -l h_rt=240:0:0  # Request  10 days runtime
#$ -l h_vmem=  20  # Request 8 * 20 = 160G total RAM
#$ -l gpu=1         # request 1 GPU
#$ -m bea
#$ -l cluster="andrena"
#$ -t 1-10

source env/bin/acivate
python main.py fit -c configs/configs_hpc.yaml -c configs/optimizer.yaml -c configs/data/combined_hpc.yaml -c configs/models/ke_dmc_adv.yaml