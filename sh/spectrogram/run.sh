#!/bin/bash
cd $HOME/Dropbox/UT/mist6/cmf/sh/spectrogram
cd $1
sh_list="./*.sh"
for sh_file in $sh_list
do
    qsub $sh_file
done
