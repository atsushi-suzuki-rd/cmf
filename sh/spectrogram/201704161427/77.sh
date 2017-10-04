#!/bin/bash
#PBS -N spectrogram_201704161427_77
#PBS -o 201704161427_77_stdout.txt
#PBS -e 201704161427_77_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704161427_77/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704161427_77/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704161427_77/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704161427_77
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704161427_77/cmf
cd /home/suzuki/output/cmf/dat/201704161427_77/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Swinging Steaks - Lost My Way/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704161427_77/json -pd /home/suzuki/output/cmf/dat/201704161427_77/pickle -ss 4 -tr 30 -te 10 -ds 2