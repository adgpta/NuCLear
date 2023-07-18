#!/bin/bash
grep -L "Finished prediction" *.out | awk -F. '{print "sbatch " $1".slurm"}' > resubmit_failed.sh
