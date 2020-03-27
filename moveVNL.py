import os

for dirname, dirnames, filenames in os.walk('/home/cc/datasets/cityscapes/leftImg8bit'):
    os.makedirs('/home/cc/datasets/cityscapes/VNL_Monocular/{}'.format(dirname.replace('/home/cc/datasets/cityscapes/leftImg8bit/', '')))

for dirname, dirnames, filenames in os.walk('/home/cc/datasets/cityscapes/leftImg8bit'):
    for filename in filenames:
        vnl_path = '/home/cc/cityscapes/VNL_Monocular/{}'.format(filename)
        new_vnl_path = '/home/cc/datasets/cityscapes/VNL_Monocular/{}/{}'.format(
            dirname.replace('/home/cc/datasets/cityscapes/leftImg8bit/', ''),
            filename.replace('leftImg8bit', 'VNL_Monocular'))
        try:
            os.rename(vnl_path, new_vnl_path)
        except FileNotFoundError as e:
            print(e)