#/bin/bash

# Output dir
output_dir="output"
rm -rf ${output_dir}
mkdir -p ${output_dir}

# Layer to optimise 
opt_layer=fc6

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt

# Get label for each unit
path_labels="misc/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

# Intiial image dir and images
init_dir=images/Init/
declare -a original_init_files=("junco.jpg" "bolete.jpg"  "lightshade.jpg" "volcano.jpg" "cardoon.jpg")

declare -a blurred_init_files=("Blur_junco.jpg" "Blur_bolete.jpg"  "Blur_lightshade.jpg" "Blur_volcano.jpg" "Blur_cardoon.jpg")

# Target categories
declare -a original_image_units=( 13 997 846 980 946 )
declare -a shifted_image_units=(946 13 997 846 980  )

declare -a conv34_image_units=( 43 163 184 1 211)
declare -a conv5_image_units=( 43 163 184 335 361 )

# Iterations
iters="1001"
#iters="10"

# Other settings
deepdream_rates="1"
deepdream_end_lr=1

generative_rates="0.2"
generative_end_lr=0.2

montage_iters="000 010 050 100 1000"

weights="99"

xy=0

#declare -a seeds=( 1 2 3 4 5 )
declare -a seeds=( 0 0 0 0 0 )

label_on=0
