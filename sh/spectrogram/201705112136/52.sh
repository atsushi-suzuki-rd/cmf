#!/bin/bash
#PBS -N spectrogram_201705112136_52
#PBS -o 201705112136_52_stdout.txt
#PBS -e 201705112136_52_stderr.txt
#PBS -l nodes=1:ppn=1:taqy
#PBS -l mem=8gb
mkdir -p /home/suzuki/output/cmf/dat/201705112136_52/cmf
mkdir -p /home/suzuki/output/cmf/dat/201705112136_52/pickle
mkdir -p /home/suzuki/output/cmf/dat/201705112136_52/json
cd /home/suzuki/Dropbox/UT/mist6/cmf
cp __init__.py /home/suzuki/output/cmf/dat/201705112136_52
cd /home/suzuki/Dropbox/UT/mist6/cmf/cmf
cp *.py /home/suzuki/output/cmf/dat/201705112136_52/cmf
cd /home/suzuki/output/cmf/dat/201705112136_52/cmf
export OMP_NUM_THREADS=1
PYTHONPATH=.. python spectrogram.py "/home/suzuki/input/MSD100/Lyndsey Ollard - Catching Up/mixture.wav" -s 0 -c 24 -w 32 -l 2000 -b 10000 -jd /home/suzuki/output/cmf/dat/201705112136_52/json -pd /home/suzuki/output/cmf/dat/201705112136_52/pickle -ss 4 -tr 30 -te 10 -ds 2