source common_setting.sh

dir="${1:-Results}"
declare -a target_dirs=( "3_complex_CBS" "4_simple_CBS_conv4")
declare -a labels=("Complex VHs in CBS" "Simple VHs in CBS")
montage_iters="000 100 1000"

result_file=result_CBS.jpg

all_files=""
for ((n=0;n<${#target_dirs[@]};++n)); do
    
    target_dir=${target_dirs[n]}
    label=${labels[n]}
    
    cd ${dir}/${target_dir}
    echo ${dir}/${target_dir}

    all_files=""
    for ((i=0;i<${#blurred_init_files[@]};++i)); do
	name=`echo ${blurred_init_files[i]} | sed 's/\.[^\.]*$//'`
    
	output_file="${name}_all.jpg"
	list_files=""
	
	for iter in ${montage_iters}; do
	    f_ini="${name}.jpg"		
	    f="${name}_${iter}.jpg"	

	    if [ "${iter}" -eq "000" ]; then
		cp "../../${init_dir}/${f_ini}" ./${f}	    
	    fi
	    
	    if [ ${i} -eq 4 ]; then
		shopt -s extglob	    
		iter_pad=${iter#+(0)}
		if [ "${iter_pad}" -eq "00" ]; then
		    iter_pad="0"
		fi
		convert $f -gravity south -splice 0x5 "spliced_${f}"
		convert "spliced_${f}" -append -gravity Center -pointsize 25 label:"iter=${iter_pad}" -bordercolor white -border 0x0 -append "labeled_${f}"
		list_files="${list_files} labeled_${f}"
	    else
		list_files="${list_files} ${f}"		    
	    fi
	    
	done
	montage ${list_files} -tile 3x1 -geometry +1+1 ${output_file}
	all_files="${all_files} ${output_file}"
    done

    montage ${all_files} -tile 1x5 -geometry +1+1 ${result_file}

    echo ${result_file}
    convert ${result_file} -gravity south -splice 0x12 "spliced_${result_file}"
    convert "spliced_${result_file}" -append -gravity Center -pointsize 32 label:"${label}" -bordercolor white -border 0x0 -append "labeled_${result_file}"

    echo "=============================="

    cd ../../
done

cd $dir


target_files="${target_dirs[0]}/labeled_${result_file} ${target_dirs[1]}/labeled_${result_file}"
echo final montage: $target_files
montage ${target_files} -tile 2x1 -geometry +10+0 CBS_pair.jpg

