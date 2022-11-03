import argparse
import os
import sys
import numpy as np
import trimesh

ALGO_TO_FUNC = {
        'naive': naive_offset,

}


def naive_offset(meshf, **kwargs):
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offset meshes, given a list of offsets')
    parser.add_argument('--indir', '-i', default='', type = str, help='path to the root dir of the data')
    parser.add_argument('--outdir', '-o', default='', type = str, help='path to the output dir')
    parser.add_argument('--meshf', '-m', default='', type = str, help='path to the mesh file')
    parser.add_argument('--offsets', '-s', type = float, nargs='+', help='list of offsets')
    parser.add_argument('--kind', '-k', default='naive', type = str, help='Offseting algorithm')

    args, unknown = parser.parse_known_args()
    print(args)