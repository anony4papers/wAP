#!/bin/bash

IMGS_DIR='city/tidim' #'bdd10k_test', 'cityscapes_test_berlin'
ATTACK_MTD='salt_numberchange' #'cw_targetclsmiss', 'noise_numberchange', 'bright_numberchange','blend_numberchange','guassian_numberchange'
#contract_numberchange,salt_numberchange

#'cw_targetclsmiss', 'noise_numberchange', 'bright_numberchange'
SQUZZEZ_MTD='bit_7'

python generate_adv.py --imgs-dir ${IMGS_DIR} --attack-mtd ${ATTACK_MTD}

python generate_squeeze.py ori --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
python generate_squeeze.py adv --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
python generate_detection_results.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
python new_cal_wap.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
python new_eval.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
