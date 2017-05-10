#!/bin/bash
#PBS -N spectrogram_201704121140_1
#PBS -o 201704121140_1_stdout
#PBS -e 201704121140_1_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704121140_1/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704121140_1/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704121140_1/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704121140_1
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704121140_1/cmf
cd /home/suzuki/output/cmf/dat/201704121140_1/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Actions - One Minute Smile/mixture.wav" -s 0 -c 24 -w  -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704121140_1/json -pd /home/suzuki/output/cmf/dat/201704121140_1/pickle -ss 4 -tr 30 -te 10 -ds 2
