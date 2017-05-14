#!/bin/bash
#PBS -N spectrogram_201705101819_55
#PBS -o 201705101819_55_stdout.txt
#PBS -e 201705101819_55_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201705101819_55/cmf
mkdir -p /home/suzuki/output/cmf/dat/201705101819_55/pickle
mkdir -p /home/suzuki/output/cmf/dat/201705101819_55/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201705101819_55
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201705101819_55/cmf
cd /home/suzuki/output/cmf/dat/201705101819_55/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Motor Tapes - Shore/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201705101819_55/json -pd /home/suzuki/output/cmf/dat/201705101819_55/pickle -ss 4 -tr 30 -te 10 -ds 2
