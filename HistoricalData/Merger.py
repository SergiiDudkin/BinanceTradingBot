#!/usr/bin/env python
import os
import sys
import numpy as np


def merge(currency):
    t_s = np.array([])
    p_s = np.array([])
    v_s = np.array([])
    folder = '{}/'.format(currency.upper())
    files = sorted([name for name in os.listdir(folder) if os.path.isfile(folder + name) and name[-4:] == '.npz'])

    if not files:
        print('No files! Nothing to merge.')
        return

    if len(files) == 1:
        print('Only single .npz file exists. Nothing to merge.')
        return

    for input_f in files:
        with np.load(folder + input_f) as data:
            t_s = np.append(t_s, data['t_s'])
            p_s = np.append(p_s, data['p_s'])
            v_s = np.append(v_s, data['v_s'])
        # print('input_f', t_s.size, p_s.size, v_s.size)

    print('First timestamp: {}, last timestamp {}.'.format(t_s[0], t_s[-1]))
    if np.all(t_s[1:] - t_s[:-1] == 900):
        print('Time consistency is Ok!')
        if t_s.size == p_s.size == v_s.size:
            print('Size consistency is Ok! Size {}'.format(t_s.size))
            for input_f in files: os.remove(folder + input_f)
            np.savez('{}{}.npz'.format(folder, int(t_s[-1])), t_s=t_s, p_s=p_s, v_s=v_s)
        else:
            print('Error: size inconsistency.')
    else:
        print('Error: time inconsistency.')


if __name__ == '__main__':
    # Settings
    args = sys.argv
    if len(args) == 2: merge(args[1])
