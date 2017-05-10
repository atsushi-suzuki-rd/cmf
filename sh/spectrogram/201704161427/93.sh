#!/bin/bash
#PBS -N spectrogram_201704161427_93
#PBS -o 201704161427_93_stdout.txt
#PBS -e 201704161427_93_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704161427_93/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704161427_93/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704161427_93/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704161427_93
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704161427_93/cmf
cd /home/suzuki/output/cmf/dat/201704161427_93/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Voelund - Comfort Lives In Belief/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704161427_93/json -pd /home/suzuki/output/cmf/dat/201704161427_93/pickle -ss 4 -tr 30 -te 10 -ds 2
