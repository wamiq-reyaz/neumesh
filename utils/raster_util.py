""" Author: Wamiq Reyaz Para
Date: 05 October 2022
Description: Utilities to save barycentric maps and face indices for meshes into 
.pkl or numpy or similar formats.
"""

import os
import sys
import pickle

import torch
import numpy as np
from tqdm import tqdm
import trimesh as trim
from rend_util import get_rays

# TODO: load a mesh
# generate a bunch of rays from view points
# perform intersection
# find barycentric coordinates of the intersection
# save those to a file


# def 