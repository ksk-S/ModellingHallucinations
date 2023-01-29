source common_setting.sh

## Target Layer [fc8 | conv3 | conv4 | conv5]
act_layer="${1:-fc8}"

## initial images [original | blurred]
init_img="${2:-original}"

## target categories [original | shifted]
target_cat="${3:-original}"

## initial images
if [ "${init_img}" = original ]; then
    init_files=(${original_init_files[@]})
elif [ "${init_img}" = blurred ]; then
    init_files=(${blurred_init_files[@]})
else
    init_files=(${original_init_files[@]})    
fi

## target categories
if [ "${target_cat}" = original ]; then
    image_units=(${original_image_units[@]})
elif [ "${target_cat}" = shifted ]; then
    image_units=(${shifted_image_units[@]})
else
    image_units=(${original_image_units[@]})
fi

## target categories for lower layers
if [ "${act_layer}" = conv5 ]; then
    image_units=("${conv5_image_units[@]}")
    xy=6    
elif [ "${act_layer}" = conv4 ]; then    
    image_units=("${conv34_image_units[@]}")
    xy=6        
elif [ "${act_layer}" = conv3 ]; then    
    image_units=("${conv34_image_units[@]}")
    xy=6
fi

# export directory
dir="${4:-export}"
cd $dir

result_file=result1.jpg
all_files=""
for ((i=0;i<${#init_files[@]};++i)); do
    name=`echo ${init_files[i]} | sed 's/\.[^\.]*$//'`
    echo ${name}    
    unit=${image_units[i]}
    
    output_file="${name}_all.jpg"
    list_files=""
    
    for iter in ${montage_iters}; do
	f_ini="${name}.jpg"		
	f="${name}_${iter}.jpg"		
	
	if [ "${iter}" -eq "000" ]; then
	    cp "../${init_dir}/${f_ini}" ./${f}	    	    
	fi
	
	if [ ${i} -eq 4 ]; then
	    shopt -s extglob	    
	    iter_pad=${iter#+(0)}
	    if [ "${iter_pad}" -eq "00" ]; then
		iter_pad="0"
	    fi
	    echo ${iter_pad}
		
	    convert $f -gravity south -splice 0x5 "spliced_${f}"
	    convert "spliced_${f}" -append -gravity Center -pointsize 25 label:"iter=${iter_pad}" -bordercolor white -border 0x0 -append "labeled_${f}"
	    list_files="${list_files} labeled_${f}"
	else
	    list_files="${list_files} ${f}"		    
	fi

    done
    montage ${list_files} -tile 5x1 -geometry +1+1 ${output_file}
    all_files="${all_files} ${output_file}"
done

montage ${all_files} -tile 1x5 -geometry +1+1 ${result_file}
echo "=============================="
