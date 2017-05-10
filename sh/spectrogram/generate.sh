#!/bin/bash
cd $HOME/Dropbox/UT/mist6/cmf/sh/spectrogram
mkdir $1
cd $1
i=0
input_dir=$HOME/input/MSD100
seed_number=0
component_max=24
loop_max=2000
base_max=1000
regularized_max=100
sampling_step=4
training_area_length=30
test_area_length=10
down_sampling_step=2
convolution_width=32
for wav_file_name in $input_dir/*/*.wav
do
    file_name=$i".sh"
    echo $file_name
    dir_name=$1"_"$i
    machine_name="taqy"
    output_dir_path="$HOME/output/cmf/dat/"$dir_name
    py_dir_path=$output_dir_path"/cmf"
    pickle_dir_path=$output_dir_path"/pickle"
    json_dir_path=$output_dir_path"/json"
    echo -e "#!/bin/bash">>$file_name
    echo -e "#PBS -N spectrogram_"$1"_"$i>>$file_name
    echo -e "#PBS -o "$1"_"$i"_stdout.txt">>$file_name
    echo -e "#PBS -e "$1"_"$i"_stderr.txt">>$file_name
    echo -e "#PBS -l nodes=1:"$machine_name>>$file_name
    echo -e "#PBS -l mem=12gb">>$file_name
    echo -e "mkdir -p "$py_dir_path>>$file_name
    echo -e "mkdir -p "$pickle_dir_path>>$file_name
    echo -e "mkdir -p "$json_dir_path>>$file_name
    echo -e "cd $HOME/Dropbox/UT/mist6/cmf">>$file_name
    echo -e "cp __init__.py "$output_dir_path>>$file_name
    echo -e "cd $HOME/Dropbox/UT/mist6/cmf/cmf">>$file_name
    echo -e "cp *.py "$py_dir_path>>$file_name
    echo -e "cd "$py_dir_path>>$file_name
    echo -e 'export OMP_NUM_THREADS=1'>>$file_name
    echo -e 'PYTHONPATH=.. python spectrogram.py "'$wav_file_name'" -s '$seed_number' -c '$component_max' -w '$convolution_width' -l '$loop_max' -b '$base_max' -jd '$json_dir_path' -pd '$pickle_dir_path' -ss '$sampling_step' -tr '$training_area_length' -te '$test_area_length' -ds '$down_sampling_step>>$file_name
    i=`expr $i + 1`
done
