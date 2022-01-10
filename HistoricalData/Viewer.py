#!/usr/bin/env python
import sys
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt


def display_price_volume(file):
    # Load data
    with np.load(file) as data:
        p_s = data['p_s']
        t_s = data['t_s']
        v_s = data['v_s']

    # Build plot
    ax0 = plt.subplot(2,1,1)
    ax1 = plt.subplot(2,1,2, sharex=ax0)
    time = np.array([datetime.fromtimestamp(t) for t in t_s])
    ax0.plot(time, p_s, color='b', linewidth=1, label='price')
    ax1.plot(time, v_s, color='r', linewidth=1, label='volume')
    ax0.legend(loc='best', fancybox=False, fontsize='medium')
    ax1.legend(loc='best', fancybox=False, fontsize='medium')
    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax0.grid()
    ax1.grid()
    ax0.label_outer()
    ax1.label_outer()
    plt.subplots_adjust(left=0.04, bottom=0.03, right=0.995, top=0.99, hspace=0)
    plt.show()


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2: display_price_volume(args[1])
