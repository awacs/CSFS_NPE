#!/bin/bash
# cousinssim="/u/scratch/h/haroldzw/CSFS/Simulation/C1/prufer_with_ghost.csfs"
cousinssim="/u/scratch/h/haroldzw/CSFS/Simulation/C1/prufer_ghost_constant.csfs"
Rscript abc_new.R --target $cousinssim --simTAG wideprior.csv100000
