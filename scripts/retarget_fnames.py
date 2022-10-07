# coding: utf-8
import os 
import argparse
import re
import shutil
 
 
def get_possible_replace_name(infile):
    """ Returns the path to be replaced. Path should start with / and contains synthetic_fur_dataset
        as a substring, which is the directory one level above
    """
    with open(infile, 'r') as fid:
        filedata = fid.read()

    
    res = re.findall('(/.*?synthetic_fur_dataset.*)/.*/', filedata)
    if res:
        return ''.join(res[0])
    else:
        raise RuntimeError("Path not found in the data file")

def rename_in_json(infile, oldpath, newpath):
    """ Opens infile and replace all occurences of old path with new path
        and writes it to the same location
    """
    with open(infile, 'r') as fid:
        filedata = fid.read()
        newdata = filedata.replace(oldpath, newpath)
    with open(infile, 'w') as fid:
        fid.write(newdata)
    return True


def get_common_parent_dir(d1, d2):
    """ Get the common parent dir. This should be the directory one level below
        synthetic_fur_dataset. A one word substring check is performed.
    """
    parent1 = os.path.abspath(os.path.join(d1, os.pardir))
    parent2 = os.path.abspath(os.path.join(d2, os.pardir))

    if (parent1) == (parent2):
        return parent1
    else:
        raise RuntimeError("Paths do not match")



# aa = get_possible_replace_name('/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/blender_textured_cube/cube_pastel_scaled_train/transforms.json')
# print(aa)
 
# with open('transforms_train.json', 'r') as f:
#     in_str = f.read()

# out_str = in_str.replace('/Volumes/GoogleDrive-111765101533328510348/My Drive/Research/cvpr23/synthetic_fur_dataset/cube_basic/', './train/')
# with open('transforms_train.json', 'w') as f:
#     f.write(out_str)
    
# with open('transforms_test.json', 'r') as f:
#     in_str = f.read()
    
# out_str = in_str.replace('/Volumes/GoogleDrive-111765101533328510348/My Drive/Research/cvpr23/synthetic_fur_dataset/cube_basic_100/', './test/')
# with open('transforms_test.json', 'w') as f:
#     f.write(out_str)
    
# get_ipython().run_line_magic('save', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', 
                        default=None,
                        help="Directory containing train images...")
    parser.add_argument('--test_dir',
                        default=None,
                        help="Directory containing test images...")
    parser.add_argument('--dry_run',
                        action='store_true',
                        help="Dry run if set, helps to check the command beforehand...")
    
    args = parser.parse_args()

    # ----------------------
    # first move the transforms.json to the parent dir
    # ----------------------
    parent_dir = get_common_parent_dir(args.train_dir, args.test_dir)
    train_transform = os.path.join(args.train_dir, 'transforms.json')
    test_transform = os.path.join(args.test_dir, 'transforms.json')

    print("Moving transform.json files to parent directory of train and test directories")

    # ---------------------------
    # Rename to transforms_train.json and transforms_test.json
    # --------------------------
    dest_file_train = os.path.join(parent_dir, 'transforms_train.json')
    dest_file_test = os.path.join(parent_dir, 'transforms_test.json')

    if not args.dry_run:
        shutil.copy(train_transform, dest_file_train)
        shutil.copy(test_transform, dest_file_test)
        print('Moved transforms.json files to ', parent_dir)
    else:
        print('Command would have moved transform.json files to ', parent_dir)


    # ---------------------------------------
    # Rename the directories
    # ---------------------------------------
    if not args.dry_run:
        os.rename(args.train_dir, os.path.join(parent_dir, 'train'))
        os.rename(args.test_dir, os.path.join(parent_dir, 'test'))
        print('Renamed old train and test directories')
    else:
        print('Command would have renamed old train and test directories')
        print('Directories:')
        print('\t', args.train_dir, '-->', os.path.join(parent_dir, 'train'))
        print('\t', args.test_dir, '-->', os.path.join(parent_dir, 'test'))



    # ----------------------------
    # Next, for each image file, we need to retarget the image names
    # Make them point relatively to the parent dir
    # ----------------------------

    print('Processing retargeting directories inside jsons...')

    train_bname = os.path.basename(args.train_dir)
    test_bname = os.path.basename(args.test_dir)

    new_names = ['./train', './test']

    for fname, new_name in zip([dest_file_train, dest_file_test], new_names):
        oldname = get_possible_replace_name(fname)
        if not args.dry_run:
            rename_in_json(fname, oldname, new_name)
            print('Replaced %s with %s in %s' % (oldname, new_name, fname))
        else:
            print('Command would have replaced %s with %s in %s' % (oldname, new_name, fname))

    print('Done!')


