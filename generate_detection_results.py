from __future__ import absolute_import
import sys
sys.path.append('/home/anony4papers/workspace/perceptron-benchmark')

from PIL import Image
import argparse
import numpy as np 
import os
import pdb
import pickle
from tqdm import tqdm
import shutil

from perceptron.utils.image import load_image, draw_letterbox
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.zoo.yolov3.model import YOLOv3

def main(args):
    imgs_dir = args.imgs_dir
    input_benign_dir = os.path.join(args.data_dir, imgs_dir, "benign")
    input_adv_dir = os.path.join(args.data_dir, imgs_dir, "adv")
    input_squz_dir = os.path.join(args.data_dir, imgs_dir, args.squeeze_type)
    input_adv_squz_dir = os.path.join(args.data_dir, imgs_dir, args.squeeze_type + '_adv')

    output_benign_dir = os.path.join(args.data_dir, imgs_dir, "results_benign")
    output_adv_dir = os.path.join(args.data_dir, imgs_dir, "results_adv")
    output_squz_dir = os.path.join(args.data_dir, imgs_dir, 'results_' + args.squeeze_type)
    output_adv_squz_dir = os.path.join(args.data_dir, imgs_dir, 'results_' + args.squeeze_type + '_adv')
    for check_dir in [output_benign_dir, output_adv_dir, output_squz_dir, output_adv_squz_dir]:
        if os.path.exists(check_dir):
            shutil.rmtree(check_dir)
        os.mkdir(check_dir)

    image_name_list = os.listdir(input_adv_dir)

    kmodel = YOLOv3()
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    for _, image_name in enumerate(tqdm(image_name_list)):
        image_name_noext = os.path.splitext(image_name)[0]

        temp_img_path_benign = os.path.join(input_benign_dir, image_name)
        temp_img_path_squz = os.path.join(input_squz_dir, image_name)
        temp_img_path_adv = os.path.join(input_adv_dir, image_name)
        temp_img_path_adv_squz = os.path.join(input_adv_squz_dir, image_name)
        try:
            print(temp_img_path_benign)
            image_benign = load_image(
                    shape=(416, 416), bounds=(0, 1),
                    fname=temp_img_path_benign,
                    absolute_path=True
            )
            image_benign_squz = load_image(
                    shape=(416, 416), bounds=(0, 1), 
                    fname=temp_img_path_squz,
                    absolute_path=True
            )
            image_adv = load_image(
                    shape=(416, 416), bounds=(0, 1), 
                    fname=temp_img_path_adv,
                    absolute_path=True
            )
            image_adv_squz = load_image(
                    shape=(416, 416), bounds=(0, 1),
                    fname=temp_img_path_adv_squz,
                    absolute_path=True
            )
        except:
            print('loading images error.')
            continue

        try:
            output_benign = model.predictions(image_benign)
            output_adv = model.predictions(image_adv)
            output_benign_squz = model.predictions(image_benign_squz)
            output_adv_squz = model.predictions(image_adv_squz)
            
            with open(os.path.join(output_benign_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_benign, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_adv_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_adv, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_squz_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_benign_squz, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_adv_squz_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_adv_squz, f, protocol=pickle.HIGHEST_PROTOCOL)

            # draw = draw_letterbox(image_benign, output_benign, (416, 416), model.class_names())
            # draw.save('./temp/{0}_benign.png'.format(image_name_noext))
            # draw = draw_letterbox(image_adv, output_adv, (416, 416), model.class_names())
            # draw.save('./temp/{0}_adv.png'.format(image_name_noext))
            # draw = draw_letterbox(image_benign_squz, output_benign_squz, (416, 416), model.class_names())
            # draw.save('./temp/{0}_benign_squz.png'.format(image_name_noext))
            # draw = draw_letterbox(image_adv_squz, output_adv_squz, (416, 416), model.class_names())
            # draw.save('./temp/{0}_adv_squz.png'.format(image_name_noext))
        
        except:
            print(image_name, ' generating error.')
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection output generation.")
    parser.add_argument('--data-dir', type=str, default='/data1/anony4papers/')
    parser.add_argument('--imgs-dir', type=str, default='bdd')#city
    parser.add_argument('--squeeze-type', type=str, default='bit_7')
    args = parser.parse_args()
    main(args)
