#!/bin/bash
#PBS -N spectrogram_201704101553_7
#PBS -o 201704101553_7_stdout
#PBS -e 201704101553_7_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704101553_7/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704101553_7/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704101553_7/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704101553_7
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704101553_7/cmf
cd /home/suzuki/output/cmf/dat/201704101553_7/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/ANiMAL - Clinic A/mixture.wav" -s 0 -c 24 -w  -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704101553_7/json -pd /home/suzuki/output/cmf/dat/201704101553_7/pickle -ss 4 -tr 30 -te 10 -ds 2