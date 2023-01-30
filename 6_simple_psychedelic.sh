#/bin/bash
layer="${1:-conv4}"

bash image_generation.sh ${layer} DCNN winner original shifted 1
