import numpy as np
from numpy.random import *
from matplotlib import pylab as plt
import matplotlib.cm as cm

def group_boxplot(ax, data, title='Title', method_facecolor=None, group_name=None, median_color='black', median_linewidth=2, caps_linewidth=1.5, whiskers_linewidth=2, interval=1, widths=0.6, patch_artist=True, y_min=None, y_max=None, xticks_fontsize=12, yticks_fontsize=12, xlabel='xlabel', ylabel='ylabel', xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=12):
    if y_min is None:
        y_min = np.min(data) - (np.max(data) - np.min(data)) * 0.05
    if y_max is None:
        y_max = np.max(data) + (np.max(data) - np.min(data)) * 0.05
    (n_groups, n_samples, n_methods) = data.shape
    if group_name is None:
        group_name = []
        for i_group in range(n_groups):
            group_name.append('group '+str(i_group))
    if method_facecolor is None:
        method_facecolor = [cm.gist_rainbow(float(i) / n_methods) for i in range(n_methods)]
    for i_group in range(n_groups):
        bp = ax.boxplot(data[i_group, :, :], positions=range(interval + (interval+n_methods) * i_group, interval + (interval+n_methods) * i_group + n_methods), widths=widths, patch_artist=patch_artist)
        for i_method in range(n_methods):
            bp['boxes'][i_method].set_facecolor(method_facecolor[i_method]) #黄色(バナナ)
            plt.setp(bp['medians'][i_method], color=median_color, linewidth=median_linewidth) #メディアン
            plt.setp(bp['caps'][2*i_method], linewidth=caps_linewidth) #はこ髭の上限の線の太さ
            plt.setp(bp['caps'][2*i_method+1], linewidth=caps_linewidth) #はこ髭の下限の線の太さ
            plt.setp(bp['whiskers'][2*i_method], linewidth=whiskers_linewidth) #boxから伸びる上の点線の太さ
            plt.setp(bp['whiskers'][2*i_method+1], linewidth=whiskers_linewidth) #boxから伸びる下の点線の太さ
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylim(y_min, y_max)
    for yticklabel in ax.get_yticklabels():
        yticklabel.set_fontsize(yticks_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    xticks = ['' for i in range(n_groups * (n_methods + interval))]
    for i_group in range(n_groups):
        xticks[interval + i_group * (n_methods + interval) + (n_methods - 1) // 2] = group_name[i_group]
    ax.set_xticks([i for i in range(n_groups * (n_methods + interval))])
    ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    for xticklabel in ax.get_xticklabels():
        xticklabel.set_fontsize(xticks_fontsize)
    ax.grid(which='both', axis='y', color='gray')
    return ax

def make_legend(ax, method_name, method_facecolor=None, linewidth=6, fontsize=6):
    line_list = []
    if method_facecolor is None:
        method_facecolor = [cm.gist_rainbow(float(i) / len(method_name)) for i in range(len(method_name))]
    for i_method in range(len(method_name)):
        line_list.append(ax.plot([1,1], color=method_facecolor[i_method], linewidth=linewidth)[0])
    ax.legend(line_list, method_name, fontsize=fontsize)
    for i_group in range(len(method_facecolor)):
        line_list[i_group].set_visible(False)
