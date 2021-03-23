import os
import random
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from tqdm import tqdm


def augment_data(root_dir, nb_copies, dynamic, uppr_lim):

    sub_dirs = tuple(os.listdir(root_dir))

    if dynamic and uppr_lim is not None:
        N = tuple(range(uppr_lim + 1))
    else:
        N = nb_copies

    for dir in sub_dirs:
        dir_path = os.path.join(root_dir, dir)
        fnames = tuple(os.listdir(dir_path))

        for fname in tqdm(fnames, desc=dir):
            name = fname.split('.')[0]
            ext = fname.split('.')[-1]
            fpath = os.path.join(dir_path, fname)
            image = cv2.imread(fpath)

            if dynamic and uppr_lim is not None:
                n = random.choice(N)
            else:
                n = N

            images = [image] * n
            try:
                aug_images = pipeline(images=images)

                for idx, img in enumerate(aug_images):
                    cv2.imwrite(dir_path + name + '_' + str(idx) + ext, img)

            except:
                pass


def split_aug_data(root_dir, split_perc, target_dir):

    sub_dirs = tuple(os.listdir(root_dir))

    for dir in sub_dirs:
        dir_path = os.path.join(root_dir, dir)
        fnames = tuple(os.listdir(dir_path))
        # TODO


def augment_init(
    root_dir,
    LinContL=0.75, LinContU=1.5,
    CropPercU=0.1,
    SatFL=0.8, SatFU=1.2,
    ScaleXL=0.9, ScaleXU=1.2, ScaleYL=0.9, ScaleYU=1.2,
    TransXL=-0.2, TransXU=0.2, TransYL=-0.2, TransYU=0.2,
    RotnL=-15, RotnU=15,
    ShearL=-8, ShearU=8,
    MotionBlurKernL=3, MotionBlurKernU=5,
    MotionBlurAngleL=0, MotionBlurAngleU=15,
    MotionBlurDirL=0.6, MotionBlurDirU=0.6,
    nb_copies=6,
    dynamic=False,
    uppr_lim=None,
    split_data=False,
    split_perc=0.0,
    target_dir=None
):

    global aug_seq

    aug_seq = iaa.Sequential([
                iaa.LinearContrast((LinContL, LinContU)),
                iaa.Crop(percent=(0, CropPercU)),
                iaa.Multiply((SatFL, SatFU), per_channel=0.2),
                iaa.Affine(
                    scale={"x": (ScaleXL, ScaleXU),
                           "y": (ScaleYL, ScaleYU)},
                    translate_percent={"x": (TransXL, TransXU),
                                       "y": (TransYL, TransYU)},
                    rotate=(RotnL, RotnU),
                    shear=(ShearL, ShearU),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                    ),
                iaa.blur.MotionBlur((MotionBlurKernL, MotionBlurKernU),
                                    (MotionBlurAngleL, MotionBlurAngleU),
                                    (MotionBlurDirL, MotionBlurDirU))
    ], random_order=True)

    global pipeline

    pipeline = iaa.WithColorspace(from_colorspace="BGR",
                                  to_colorspace="RGB",
                                  children=aug_seq)

    augment_data(root_dir, nb_copies, dynamic=False, uppr_lim=None)

    if split_data and split_perc > 0 and (target_dir is not None):
        split_aug_data(root_dir, split_perc, target_dir)
