import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pandas as pd

NUM_REAL_TEST_IMAGES = 6
NUM_FAKE_TEST_IMAGES = 39
NUM_REAL_TRAIN_IMAGES = 6

for ANALYSIS_AXIS in range(3):
    SLICE_IDXS = [18, 23, 25, 32] if ANALYSIS_AXIS < 2 else [13, 23, 52, 75]
    for SLICE_IDX in SLICE_IDXS:
        commands = []
        for i in range(NUM_REAL_TEST_IMAGES):
            commands.append((f"python SIFID/sifid_score.py "
                             f"--path2real neurips_images/{SLICE_IDX}/singan/sifid/real_{i}/ct/{ANALYSIS_AXIS} "
                             f"--path2fake neurips_images/{SLICE_IDX}/singan/sifid/fake/ct/{ANALYSIS_AXIS} "
                             f"--save_file singan_sifid_ct_{SLICE_IDX}_{i}_{ANALYSIS_AXIS} --images_suffix png --gpu 1"))
        for i in range(NUM_REAL_TEST_IMAGES):
            commands.append((f"python SIFID/sifid_score.py "
                             f"--path2real neurips_images/{SLICE_IDX}/singan/sifid/real_{i}/pet/{ANALYSIS_AXIS} "
                             f"--path2fake neurips_images/{SLICE_IDX}/singan/sifid/fake/pet/{ANALYSIS_AXIS} "
                             f"--save_file singan_sifid_pet_{SLICE_IDX}_{i}_{ANALYSIS_AXIS} --images_suffix png --gpu 1"))
        
        for i in range(NUM_REAL_TEST_IMAGES):
            commands.append((f"python SIFID/sifid_score.py "
                             f"--path2real neurips_images/{SLICE_IDX}/singan/sifid/real_{i}/ct/{ANALYSIS_AXIS} "
                             f"--path2fake neurips_images/{SLICE_IDX}/cyclegan/sifid/fake/ct/{ANALYSIS_AXIS} "
                             f"--save_file cyclegan_sifid_ct_{SLICE_IDX}_{i}_{ANALYSIS_AXIS} --images_suffix png --gpu 1"))
        
        for i in range(NUM_REAL_TEST_IMAGES):
            commands.append((f"python SIFID/sifid_score.py "
                             f"--path2real neurips_images/{SLICE_IDX}/singan/sifid/real_{i}/pet/{ANALYSIS_AXIS} "
                             f"--path2fake neurips_images/{SLICE_IDX}/cyclegan/sifid/fake/pet/{ANALYSIS_AXIS} "
                             f"--save_file cyclegan_sifid_pet_{SLICE_IDX}_{i}_{ANALYSIS_AXIS} --images_suffix png --gpu 1"))
        
        for command in commands:
            subprocess.run(command.split())