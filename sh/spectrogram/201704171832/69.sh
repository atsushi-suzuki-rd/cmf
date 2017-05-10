#!/bin/bash
#PBS -N spectrogram_201704171832_69
#PBS -o 201704171832_69_stdout.txt
#PBS -e 201704171832_69_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704171832_69/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704171832_69/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704171832_69/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704171832_69
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704171832_69/cmf
cd /home/suzuki/output/cmf/dat/201704171832_69/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Signe Jakobsen - What Have You Done To Me/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704171832_69/json -pd /home/suzuki/output/cmf/dat/201704171832_69/pickle -ss 4 -tr 30 -te 10 -ds 2
