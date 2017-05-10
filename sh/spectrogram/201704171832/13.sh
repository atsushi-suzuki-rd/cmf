#!/bin/bash
#PBS -N spectrogram_201704171832_13
#PBS -o 201704171832_13_stdout.txt
#PBS -e 201704171832_13_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704171832_13/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704171832_13/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704171832_13/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704171832_13
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704171832_13/cmf
cd /home/suzuki/output/cmf/dat/201704171832_13/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Bill Chudziak - Children Of No-one/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704171832_13/json -pd /home/suzuki/output/cmf/dat/201704171832_13/pickle -ss 4 -tr 30 -te 10 -ds 2
