#!/bin/bash
#PBS -N spectrogram_201704131825_0
#PBS -o 201704131825_0_stdout
#PBS -e 201704131825_0_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704131825_0/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704131825_0/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704131825_0/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704131825_0
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704131825_0/cmf
cd /home/suzuki/output/cmf/dat/201704131825_0/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Actions - Devil's Words/mixture.wav" -s 0 -c 24 -w  -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704131825_0/json -pd /home/suzuki/output/cmf/dat/201704131825_0/pickle -ss 4 -tr 30 -te 10 -ds 2
