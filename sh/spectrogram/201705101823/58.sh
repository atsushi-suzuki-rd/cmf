#!/bin/bash
#PBS -N spectrogram_201705101823_58
#PBS -o 201705101823_58_stdout.txt
#PBS -e 201705101823_58_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201705101823_58/cmf
mkdir -p /home/suzuki/output/cmf/dat/201705101823_58/pickle
mkdir -p /home/suzuki/output/cmf/dat/201705101823_58/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201705101823_58
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201705101823_58/cmf
cd /home/suzuki/output/cmf/dat/201705101823_58/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/North To Alaska - All The Same/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201705101823_58/json -pd /home/suzuki/output/cmf/dat/201705101823_58/pickle -ss 4 -tr 30 -te 10 -ds 2
