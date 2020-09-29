from __future__ import absolute_import
import sys
sys.path.append('/home/anony4papers/workspace/perceptron-benchmark')

from PIL import Image
import argparse
import numpy as np 
import os
import shutil
import pdb
from tqdm import tqdm

from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.utils.image import load_image
from perceptron.benchmarks.carlini_wagner import CarliniWagnerLinfMetric
from perceptron.benchmarks.additive_noise import AdditiveGaussianNoiseMetric,AdditiveUniformNoiseMetric
from perceptron.benchmarks.brightness import BrightnessMetric
from perceptron.benchmarks.blended_noise import BlendedUniformNoiseMetric
from perceptron.benchmarks.gaussian_blur import GaussianBlurMetric
from perceptron.benchmarks.contrast_reduction import ContrastReductionMetric
from perceptron.benchmarks.motion_blur import MotionBlurMetric
from perceptron.benchmarks.rotation import RotationMetric
from perceptron.benchmarks.salt_pepper import SaltAndPepperNoiseMetric
from perceptron.benchmarks.spatial import SpatialMetric
from perceptron.utils.criteria.detection import TargetClassMiss, TargetClassNumberChange

DEBUG = False

def main(args):
    imgs_dir = args.imgs_dir
    input_dir = os.path.join(args.data_dir, imgs_dir, "benign")
    if not DEBUG:
        output_dir = os.path.join(args.data_dir, imgs_dir, "adv")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    image_name_list = os.listdir(input_dir)

    kmodel = YOLOv3()
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    if args.attack_mtd == 'cw_targetclsmiss':
        attack = CarliniWagnerLinfMetric(model, criterion=TargetClassMiss(2))
    elif args.attack_mtd == 'noise_numberchange':
        attack = AdditiveGaussianNoiseMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'bright_numberchange':
        attack = BrightnessMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'blend_numberchange':
        attack = BlendedUniformNoiseMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'guassian_numberchange':
        attack = GaussianBlurMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'contract_numberchange':
        attack = ContrastReductionMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'salt_numberchange':
        attack = SaltAndPepperNoiseMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'spatial_numberchange':
        attack = SpatialMetric(model, criterion=TargetClassNumberChange(-1))
    else:
        raise ValueError('Invalid attack method {0}'.format( args.attack_mtd))

    for _, image_name in enumerate(tqdm(image_name_list)):

        temp_img_path_benign = os.path.join(input_dir, image_name)
        #temp_img_path_benign = '/data1/anony4papers/bdd/benign/'+image_name
        image_benign = load_image(
                shape=(416, 416), bounds=(0, 1),
                fname=temp_img_path_benign,
                absolute_path=True
        )

        try:
            annotation_ori = model.predictions(image_benign)
            #print(annotation_ori)
            if args.attack_mtd == 'cw_targetclsmiss':
                image_adv_benign = attack(image_benign, binary_search_steps=1, unpack=True)
            else:
                image_adv_benign = attack(image_benign, annotation_ori,epsilons=255, unpack=True)
        except:
            print('Attack failed.')
            continue

        try: 
            image_adv_benign_pil = Image.fromarray((image_adv_benign * 255).astype(np.uint8))
        except:
            continue
        if not DEBUG:
            try:
                image_adv_benign_pil.save(os.path.join(output_dir, image_name))
            except:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squeeze image generation.")
    parser.add_argument('--data-dir', type=str, default='/data1/anony4papers/')
    parser.add_argument('--imgs-dir', type=str, default='bdd')#bdd
    parser.add_argument('--attack-mtd', type=str, default='bright_numberchange')
    args = parser.parse_args()
    main(args)
