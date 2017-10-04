#!/bin/bash
#PBS -N spectrogram_201703311719_5
#PBS -o 201703311719_5_stdout
#PBS -e 201703311719_5_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201703311719_5/cmf
mkdir -p /home/suzuki/output/cmf/dat/201703311719_5/pickle
mkdir -p /home/suzuki/output/cmf/dat/201703311719_5/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201703311719_5
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201703311719_5/cmf
cd /home/suzuki/output/cmf/dat/201703311719_5/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Angela Thomas Wade - Milk Cow Blues/mixture.wav" -s 0 -c 24 -w  -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201703311719_5/json -pd /home/suzuki/output/cmf/dat/201703311719_5/pickle -ss 4 -tr 30 -te 10 -ds 2