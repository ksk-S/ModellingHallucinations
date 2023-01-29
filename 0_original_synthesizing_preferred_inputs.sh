#/bin/bash

# Input images
init_dir=images/Examples/

declare -a init_files=(
    "red_pepper.jpg"
    "GoldFish.jpg"
    "SewingMachine.jpg"
    "DialPhone.jpg"
    "birdhouse.jpg"
)
#init_file=images/Black.jpg

# Output dir
output_dir="output"
rm -rf ${output_dir}
mkdir -p ${output_dir}

# Target categories
declare -a image_units=(1 786 528 448 945)	       

# Get label for each unit
path_labels="misc/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer=fc6
act_layer=fc8

xy=0

# Hyperparam settings for visualizing AlexNet
iters="300"
weights="99"

#rates="8.0"
#end_lr=1e-10

# clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt

#using deep dream algorithim to update the image or not
deep_dream=0

#using L2 norm maximsiaiton rather than fixed unit
act_mode='l2norm'
#act_mode='fixed'
#act_mode='winner'

#raw image
raw_ini_img=0

if [ "${deep_dream}" -eq "1" ]; then
    rates="1"
    end_lr=1

    jitter=4
    rotation=5

    gen_mode="dream"
else
    rates="0.1"
    end_lr=0.01
    
    jitter=0
    rotation=0

    gen_mode="gen"    
fi

# Stats mode
stats=0

label_on=0

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
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
	#for seed in {0..8}; do
	
	for n_iters in ${iters}; do
	    for w in ${weights}; do
		for lr in ${rates}; do
		    
		    L2="0.${w}"
		    
		    # Optimize images maximizing fc8 unit
		    python3 ./act_max_dream.py \
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
			    --stats ${stats}
		    
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
