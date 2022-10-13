import os 
import imageio
from glob import glob
import argparse
from natsort import natsorted

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a movie from a folder of images')
    parser.add_argument('-f', '--foldername', default=None)
    parser.add_argument('-o', '--outputfilename', default=None)
    args = parser.parse_args()

    foldername = args.foldername
    output_it2 = args.outputfilename

    if not os.path.isdir(foldername):
        print("Folder {} does not exist".format(foldername))
        exit(1)
    if os.path.isfile(output_it2):
        print("Outputfile {} does already exist".format(output_it2))

    filenames = glob(foldername + "/*.npy")
    filnames = natsorted(filenames)
    # filenames.sort()

    with imageio.get_writer(output_it2, mode='I', fps=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("Created movie {} from {} images".format(output_it2, len(filenames)))