#!/bin/bash
#PBS -N spectrogram_201704171832_44
#PBS -o 201704171832_44_stdout.txt
#PBS -e 201704171832_44_stderr.txt
#PBS -l nodes=1:taqy
#PBS -l mem=12gb
mkdir -p /home/suzuki/output/cmf/dat/201704171832_44/cmf
mkdir -p /home/suzuki/output/cmf/dat/201704171832_44/pickle
mkdir -p /home/suzuki/output/cmf/dat/201704171832_44/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201704171832_44
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201704171832_44/cmf
cd /home/suzuki/output/cmf/dat/201704171832_44/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Johnny Lokke - Whisper To A Scream/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 1000 -jd /home/suzuki/output/cmf/dat/201704171832_44/json -pd /home/suzuki/output/cmf/dat/201704171832_44/pickle -ss 4 -tr 30 -te 10 -ds 2
