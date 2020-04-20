import argparse
import glob
import tensorflow as tf
import pandas as pd
import os.path
from tqdm import tqdm
import math

import matplotlib.pyplot as plt


def plot_range(metric, title, use_max=True):
    y1 = metric.transpose().min()
    y2 = metric.transpose().max()

    fig = plt.figure(figsize=(16, 14))
    plt.suptitle(title, fontsize=14)

    key_list = [[x for x in metric.keys() if 'rgb/' in x]]
    key_list.append([x for x in metric.keys() if 'rgb_pt' in x])
    key_list.append([x for x in metric.keys() if 'rgbd' in x])
    key_list.append([x for x in metric.keys() if 'synthetic' in x])
    key_list.append([x for x in metric.keys() if 'latefusion' in x])
    key_list.append([x for x in metric.keys() if 'midfusion' in x])

    n = len(metric.keys())
    cols = max([len(row) for row in key_list])
    rows = len(key_list)

    for i, row in enumerate(key_list):
        row.sort()

        for j, key in enumerate(row):
            x = i*cols + j
            ax = plt.subplot(rows, cols, x+1)

            plt.plot(metric.index, metric[key], 'k', linewidth=1)
            plt.fill_between(metric.index, y1, y2, color='b', alpha=0.2)
            plt.title(key)

            style = dict(size=10, color='black')
            if use_max:
                max_idx = metric[key].argmax()
                max_val = metric[key].max()
                ax.text(max_idx, max_val, "{:0.2f}".format(max_val), **style)
            else:
                min_idx = metric[key].argmin()
                min_val = metric[key].min()
                ax.text(min_idx, min_val, "{:0.2f}".format(min_val), **style)

            if j > 0:
                ax.yaxis.set_visible(False)
            if i < rows-1:
                ax.xaxis.set_visible(False)

    plt.show()

    return fig


def save_tensorboard_to_csv(root, filelist):
    sess = tf.InteractiveSession()

    df_dict = {}
    with sess.as_default():
        for fn in tqdm(filelist):
            index_list = {}
            output_fn = os.path.join(os.path.split(fn)[0], 'tensorboard.csv')
            try:
                for e in tf.train.summary_iterator(fn):
                    if e.step <= 200:
                        for v in e.summary.value:
                            if v.tag != "train/total_loss_iter":
                                if e.step in index_list:
                                    index_list[e.step][v.tag] = v.simple_value
                                else:
                                    index_list[e.step] = {v.tag: v.simple_value}
            except Exception as e:
                print("{}: {}".format(fn, e.message))

            finally:
                fn_key = os.path.split(fn)[0].replace(root, "")
                df_dict[fn_key] = pd.DataFrame(index_list.values(), index=index_list.keys())

                if fn_key == 'rgb/experiment_1':
                    df_dict['rgb/experiment_1'] = pd.concat([df_dict['rgb/experiment_0'], df_dict['rgb/experiment_1']],
                                                            axis=0)
                    del df_dict['rgb/experiment_0']
                elif fn_key == 'rgbd/2020-03-06':
                    df_dict['rgbd/2020-03-06'] = pd.concat([df_dict['rgbd/2020-03-06'], df_dict['rgbd/2020-03-09']],
                                                            axis=0)
                    del df_dict['rgbd/2020-03-09']
                df_dict[fn_key].to_csv(output_fn)

    m_iou = pd.concat([df_dict[key]["val/mIoU"] for key in df_dict.keys()], axis=1, keys=df_dict.keys())
    fw_iou = pd.concat([df_dict[key]["val/fwIoU"] for key in df_dict.keys()], axis=1, keys=df_dict.keys())
    loss = pd.concat([df_dict[key]["train/total_loss_epoch"] for key in df_dict.keys()], axis=1, keys=df_dict.keys())
    acc_class = pd.concat([df_dict[key]["val/Acc_class"] for key in df_dict.keys()], axis=1, keys=df_dict.keys())

    figs = []
    figs.append(plot_range(m_iou, "Mean IOU"))
    figs.append(plot_range(fw_iou, "Frequency Weighted IOU"))
    figs.append(plot_range(loss, "Loss", use_max=False))
    figs.append(plot_range(acc_class, "Class averaged accuracy"))

    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(root, "{}.png".format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()

    filelist = glob.glob(os.path.join(args.root, "*", "*", "*events.out.tfevents*"))
    save_tensorboard_to_csv(args.root, filelist)