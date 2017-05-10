#!/bin/bash
#PBS -N spectrogram_201704151303_64
#PBS -o 201704151303_64_stdout
#PBS -e 201704151303_64_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704151303_64/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704151303_64/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704151303_64/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704151303_64
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704151303_64/cmf
cd /home/suzuki/output/cmf/dat/201704151303_64/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Remember December - C U Next Time/mixture.wav" -s 0 -c 24 -w  -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704151303_64/json -pd /home/suzuki/output/cmf/dat/201704151303_64/pickle -ss 4 -tr 30 -te 10 -ds 2
