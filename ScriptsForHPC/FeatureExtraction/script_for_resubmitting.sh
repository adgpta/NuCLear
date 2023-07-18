#!/bin/bash
ls *.csv|cut -d. -f1 > finished
ls *.slurm|cut -d. -f1 > all
squeue|grep "single"|awk '{print $1}'|xargs -L1 scontrol show job|grep "JobName="|cut -d "=" -f3|cut -d "." -f1 > running
cat running >> finished
grep -v -f finished all|awk '{print $1}'|xargs -i echo sbatch {}.tif.slurm > resubmit_failed.sh
