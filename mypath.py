class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '../datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '../datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '../datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '../datasets/coco/'
        elif dataset == 'sunrgbd':
            return '../datasets/SUNRGBD/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
