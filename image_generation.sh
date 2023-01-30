#/bin/bash
source common_setting.sh

## Target Layer [fc8 | conv3 | conv4 | conv5]
act_layer="${1:-fc8}"

## Generation Type [DGN | DCNN]
gen_type="${2:-DGN}"

## Select Error Function [winner, l2norm, fixed]
act_mode="${3:-winner}" 

## initial images [original | blurred]
init_img="${4:-original}"

## target categories [original | shifted]
target_cat="${5:-original}"

## Export images at artbirary iterations in 'export' folder
export="${6:-0}"

## Export mode [exp | validation | interview]
export_mode="${7:-exp}"

if [ "${export_mode}" = validation ]; then
    source validation_setting.sh
fi

## Debug mode: create images for each iteration in Debug/ folder
debug=0

## Stats mode: export information in stats/ folder
stats=0


## DGN-AM or DCNN-AM
if [ "${gen_type}" = DGN ]; then
    deep_dream=0
elif [ "${gen_type}" = DCNN ]; then
    deep_dream=1
fi

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

if [ "${deep_dream}" -eq "1" ]; then
    raw_ini_img=0
    rates=${deepdream_rates}
    end_lr=${deepdream_end_lr}

    jitter=4
    rotation=5

    gen_mode="DCNN"
else
    raw_ini_img=1
    rates=${generative_rates}
    end_lr=${generative_end_lr}
    
    jitter=0
    rotation=0

    gen_mode="DGN"    
fi

if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

if [ "${export}" -eq "1" ]; then
    rm -rf export    
    mkdir export
fi

list_files=""
# Sweeping across hyperparams
for ((i=0;i<${#init_files[@]};++i)); do
    unit=${image_units[i]}
    init_file=${init_files[i]}
    
    # Get label for each unit
    label_1=`echo ${labels[unit]} | cut -d "," -f 1 | cut -d " " -f 2`
    label_2=`echo ${labels[unit]} | cut -d "," -f 1 | cut -d " " -f 3`
    label="${unit}: ${label_1} ${label_2}"
    
    for seed in {0..0}; do
	for n_iters in ${iters}; do
	    for w in ${weights}; do
		for lr in ${rates}; do
		    
		    L2="0.${w}"
		    
		    # Optimize images maximizing fc8 unit
		    python3 ./act_max2.py \
			    --act_layer ${act_layer} \
			    --opt_layer ${opt_layer} \
			    --unit ${unit} \
			    --xy ${xy} \
			    --n_iters ${n_iters} \
			    --start_lr ${lr} \
			    --end_lr ${end_lr} \
			    --L2 ${L2} \
			    --seed ${seed} \
			    --clip ${clip} \
			    --bound ${bound_file} \
			    --debug ${debug} \
			    --output_dir ${output_dir} \
			    --init_file ${init_dir}${init_file} \
			    --deep_dream ${deep_dream} \
 			    --act_mode ${act_mode} \
			    --jitter ${jitter} \
			    --rotation ${rotation} \
			    --raw_ini_img ${raw_ini_img} \
			    --stats ${stats} \
		            --export ${export} \
			    --export_mode ${export_mode}
		    
		    # Add a category label to each image
		    unit_pad=`printf "%04d" ${unit}`
		    n_iters_pad=`printf "%02d" ${n_iters}`	  
		    lr_pad=`printf "%.1f" ${lr}`
		    f=${output_dir}/${gen_mode}_${act_mode}_${act_layer}_${unit_pad}_${init_file}_${n_iters_pad}_${L2}_${lr_pad}__${seed}.jpg
		    #echo $f
		    if [ "${label_on}" -eq "1" ]; then	  
			convert $f -gravity south -splice 0x10 $f
			convert $f -append -gravity Center -pointsize 30 label:"$label" -bordercolor white -border 0x0 -append $f
		    fi
		    list_files="${list_files} ${f}"
		done
	    done
	done
    done
done


# Make a collage
output_file=${output_dir}/example1.jpg

montage ${list_files} -tile 5x1 -geometry +1+1 ${output_file}
convert ${output_file} -trim ${output_file}
echo "=============================="
echo "Result of example 1: [ ${output_file} ]"
