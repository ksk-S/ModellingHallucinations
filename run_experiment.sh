#/bin/bash
dir=Results
rm -rf ${dir}
mkdir ${dir}

# veridical
label=1_veridical
bash image_generation.sh fc8 DGN winner original original 1
bash scripts/make_montage.sh fc8 original original
mv export ${dir}/${label}
cp ${dir}/${label}/result1.jpg ${dir}/${label}.jpg

# complex neuro
label=2_complex_neuro
bash image_generation.sh fc8 DGN fixed original shifted 1
bash scripts/make_montage.sh fc8 original shifted
mv export ${dir}/${label}
cp ${dir}/${label}/result1.jpg ${dir}/${label}.jpg

# complex CBS
label=3_complex_CBS
bash image_generation.sh fc8 DGN fixed blurred shifted 1
bash scripts/make_montage.sh fc8 blurred shifted
mv export ${dir}/${label}
cp ${dir}/${label}/result1.jpg ${dir}/${label}.jpg

# simple CBS
label=4_simple_CBS
options="conv3 conv4"
for option in ${options}; do
    bash image_generation.sh ${option} DGN winner blurred shifted 1
    bash scripts/make_montage.sh ${option} blurred shifted
    mv export ${dir}/${label}_${option}
    cp ${dir}/${label}_${option}/result1.jpg ${dir}/${label}_${option}.jpg
done

# complex psychdelic
label=5_complex_psychedelic
bash image_generation.sh fc8 DCNN l2norm original shifted 1
bash scripts/make_montage.sh original shifted
mv export ${dir}/${label}
cp ${dir}/${label}/result1.jpg ${dir}/${label}.jpg

# simple psychedelic
label=6_simple_psychedelic
options="conv3 conv4"
for option in ${options}; do
    bash image_generation.sh ${option} DCNN winner original shifted 1
    bash scripts/make_montage.sh ${option} original shifted     
    mv export ${dir}/${label}_${option}
    cp ${dir}/${label}_${option}/result1.jpg ${dir}/${label}_${option}.jpg
done

bash scripts/make_montage_pair_CBS.sh ${dir}

