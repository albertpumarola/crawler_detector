from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

def plot_bb(img, pose, label=None, display_bb=True):
    '''
    Plot bounding box on top of image
    :param img: HxWx3
    :param pose: 4
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    # display bb
    if display_bb:
        u, v, h, w = pose[0], pose[1], pose[2] - pose[0], pose[3] - pose[1]
        ax.add_patch(patches.Rectangle((v, u), w, h, fill=False, edgecolor='green', linewidth=3))
        ax.add_patch(patches.Circle((pose[1], pose[0]), fill=True, color='red'))
        ax.add_patch(patches.Circle((pose[3], pose[2]), fill=True, color='blue'))

    # display label
    if label is not None:
        color = 'green' if display_bb else 'red'
        ax.text(0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                color=color, fontsize=60)

    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def plot_overlay_attention(image, attention,
                      alpha=0.3, cmap='jet', uv=None, prob=None):
    """
    input: image: HxWx3, attention: HxW
    usage:
    image_summary = tfplot.summary.plot("plot_summary", utils.overlay_attention, [attention,image])
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    H, W = attention.shape
    ax.imshow(image, extent=[0, H, 0, W])
    # heatmap = ax.imshow(attention, cmap=cmap,
    #                     alpha=alpha, extent=[0, H, 0, W], vmin=np.min(attention), vmax=np.max(attention))
    heatmap = ax.imshow(attention, cmap=cmap,
                        alpha=alpha, extent=[0, H, 0, W], vmin=0, vmax=1)
    if prob is not None:
        ax.text(0.5, 0.9, prob, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                color='red', fontsize=30)

    if uv is not None:
        x, y = uv[0], H-uv[1]
        ax.add_patch(patches.Circle((x, y), 5, color='red'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(heatmap, cax=cax)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def plot_center(image, uv, prob=None):
    """
    input: image: HxWx3
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    ax.imshow(image)

    x, y = uv[1], uv[0]
    ax.add_patch(patches.Circle((x, y), 5, color='red'))

    if prob is not None:
        ax.text(0.5, 0.9, prob, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                color='red', fontsize=30)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data