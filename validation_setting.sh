#/bin/bash

# validation initial images
init_dir=images/Validation/

declare -a original_init_files=("01_sewingmachine.jpg" "02_goldfish.jpg" "03_dialphone.jpg" "04_redpepper.jpg" "05_birdhouse.jpg" "06_junco.jpg" "07_bolete.jpg"  "08_lightshade.jpg" "09_volcano.jpg" "10_ladybug.jpg" "11_cock.jpg" "12_balloons.jpg" "13_baseball.jpg" "14_pineapple.jpg" "15_waterbuffalo.jpg" "16_cardoon.jpg" "17_boathouse.jpg" "18_buckets.jpg" "19_flamingo.jpg" "20_tibettanterrier.jpg" "21_safetypin.jpg" "22_container_ship.jpg" "23_patas.jpg" "24_daisy.jpg" "25_burger.jpg" "26_jackolantern.jpg" "27_starfish.jpg" "28_candle.jpg" "29_jellyfish.jpg" "30_wool.jpg"  "31_skunk.jpg" "32_motorscooter.jpg" )

declare -a original_image_units=(786 1 528 945 448 13 997 846 980 301 7 417 429 953 346 946 449 463 130 200 772 510 371 985 933 607 327 470 107 911 361 670)
declare -a shifted_image_units=(670 786 1 528 945 448 13 997 846 980 301 7 417 429 953 346 946 449 463 130 200 772 510 371 985 933 607 327 470 107 911 361 )

### Simple cateogories
declare -a conv34_image_units=( 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211 43 163 184 1 211)
declare -a conv5_image_units=( 43 163 184 335 361 )

# Other settings
iters="1002"

