#/bin/bash

# Generate images for the image selection task

dir=Validations
rm -rf ${dir}
mkdir ${dir}

label=1_veridical
bash image_generation.sh fc8 DGN winner original original 1 validation
mv export ${dir}/${label}

label=2_complex_neuro_wta
bash image_generation.sh fc8 DGN winner original shifted 1 validation
mv export ${dir}/${label}

label=3_complex_neuro_fixed
bash image_generation.sh fc8 DGN fixed original shifted 1 validation
mv export ${dir}/${label}

label=4_complex_psyche_wta
bash image_generation.sh fc8 DCNN wta original shifted 1 validation
mv export ${dir}/${label}

label=5_complex_psyche_fixed
bash image_generation.sh fc8 DCNN wta original shifted 1 validation
mv export ${dir}/${label}

label=6_simple_neuro_conv3
bash image_generation.sh conv3 DGN winner original shifted 1 validation
mv export ${dir}/${label}

label=7_simple_neuro_conv4
bash image_generation.sh conv4 DGN winner original shifted 1 validation
mv export ${dir}/${label}

label=8_simple_psyche_conv3
bash image_generation.sh conv3 DCNN winner original shifted 1 validation
mv export ${dir}/${label}

label=8_simple_psyche_conv4
bash image_generation.sh conv4 DCNN winner original shifted 1 validation
mv export ${dir}/${label}




