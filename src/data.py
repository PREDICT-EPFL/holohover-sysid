# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from copy import deepcopy
import numpy as np
import torch
import pandas as pd

from src.params import Params
from src.utils import read_mcap_file

class Data():
    def __init__(self, params: Params) -> None:
        """
        Initialization of Data instance
        Args:
            series_name: name of experiment series to load, str
        """
        # public variables
        self.exps = []
        self.series_name = params['data']['experiment']

        # private variables
        self._x = {}
        self._dx = {}
        self._ddx = {}
        self._tx = {}
        self._u = {}
        self._motor_speed = {}
        self._tu = {}

    def get(self, names):
        """
        Get privat data
        Args:
            names: list of data name, list of str
        Returns:
            data: list of data, list of dict
        """
        data = []
        for name in names:
            if name == 'x':
                data.append(deepcopy(self._x)) 
            if name == 'dx':
                data.append(deepcopy(self._dx))
            if name == 'ddx':
                data.append(deepcopy(self._ddx))
            if name == 'tx':
                data.append(deepcopy(self._tx))
            if name == 'u':
                data.append(deepcopy(self._u))
            if name == 'motor_speed':
                data.append(deepcopy(self._motor_speed))
            if name == 'tu':
                data.append(deepcopy(self._tu))
        return data

    def set(self, names, data):
        """
        Set privat data
        Args:
            names: list of data name, list of str
            data: list of data, list of dict
        """
        for name, d in zip(names, data):
            if name == 'x':
                self._x = deepcopy(d)
            if name == 'dx':
                self._dx = deepcopy(d)
            if name == 'ddx':
                self._ddx = deepcopy(d)
            if name == 'tx':
                self._tx = deepcopy(d)
            if name == 'u':
                self._u = deepcopy(d)
            if name == 'motor_speed':
                self._motor_speed = deepcopy(d)
            if name == 'tu':
                self._tu = deepcopy(d)

    def delete(self, names):
        """
        Delete privat data
        Args:
            names: list of data name, list of str
        """
        for name in names:
            if name == 'x':
                del self._x
            if name == 'dx':
                del self._dx
            if name == 'ddx':
                del self._ddx
            if name == 'tx':
                del self._tx
            if name == 'u':
                del self._u
            if name == 'motor_speed':
                del self._motor_speed
            if name == 'tu':
                del self._tu

    def shift(self, shift, names):
        """
        Shift data back in time: if shift=2 and unshifted_data=[0,0,0,1,2,3] then shifted_data=[0,1,2,3]
        All data that is not shifted is cropped at the end st. all data has the same length
        Args:
            shift: nb. of indices to shift, int
            names: list of names to shif
        """
        assert shift>0, f'Shift must be greater than zero, but shift={shift}'
        not_shifted_names = ['x', 'dx', 'ddx', 'tx', 'u', 'motor_speed', 'tu']

        for name in names:
            not_shifted_names.remove(name)

            for exp in self.exps:
                if name == 'x':
                    self._x[exp] = self._x[exp][shift:,:]
                if name == 'dx':
                    self._dx[exp] = self._dx[exp][shift:,:]
                if name == 'ddx':
                    self._ddx[exp] = self._ddx[exp][shift:,:]
                if name == 'tx':
                    self._tx[exp] = self._tx[exp][shift:] - self._tx[exp][shift]
                if name == 'u':
                    self._u[exp] = self._u[exp][shift:,:]
                if name == 'motor_speed':
                    self._motor_speed[exp] = self._motor_speed[exp][shift:,:]
                if name == 'tu':
                    self._tu[exp] = self._tu[exp][shift:] - self._tu[exp][shift]

        for name in not_shifted_names:
            if '_'+name not in self.__dict__.keys():
                continue

            for exp in self.exps:
                if name == 'x' and len(self._x)>0:
                    self._x[exp] = self._x[exp][:-shift,:]
                if name == 'dx' and len(self._dx)>0:
                    self._dx[exp] = self._dx[exp][:-shift,:]
                if name == 'ddx' and len(self._ddx)>0:
                    self._ddx[exp] = self._ddx[exp][:-shift,:]
                if name == 'tx' and len(self._tx)>0:
                    self._tx[exp] = self._tx[exp][:-shift]
                if name == 'u' and len(self._u)>0:
                    self._u[exp] = self._u[exp][:-shift,:]
                if name == 'motor_speed' and len(self._motor_speed)>0:
                    self._motor_speed[exp] = self._motor_speed[exp][:-shift,:]
                if name == 'tu' and len(self._tu)>0:
                    self._tu[exp] = self._tu[exp][:-shift]

    def save(self):
        """
        Saves data to .pt file
        """
        data = {
            'exps': self.exps,
            'T': {},
            'X': {},
            'U': {},
            'dX': {},
        }

        for exp in self.exps:
            data['T'][exp] = torch.from_numpy(self._tx[exp])
            data['X'][exp] = torch.from_numpy(np.concatenate((self._x[exp], self._dx[exp]), axis=1))
            data['U'][exp] = torch.from_numpy(self._u[exp])
            data['dX'][exp] = torch.from_numpy(np.concatenate((self._dx[exp], self._ddx[exp]), axis=1))

        torch.save(data, os.path.join('experiments', self.series_name, 'data.pt'))

    def convert_mcap_to_csv(self):
        
        # loop through all sub folders of the defined experiment series
        for root, _, files in os.walk(os.path.join('experiments', self.series_name)):
            # loop through all files in subfolder
            for f in files:
                if f.find('.mcap') != -1:
                    topic_mapping = {
                        '/drone/control': 'drone_control',
                        '/optitrack/drone/pose': 'optitrack_drone_pose',
                    }

                    file_path = os.path.join(root, f)

                    topic_msgs = read_mcap_file(file_path, topic_mapping.keys())
                    dir = os.path.dirname(file_path)

                    for topic, msgs in topic_msgs.items():
                        df = pd.DataFrame(msgs)
                        df.to_csv(os.path.join(dir, f'{topic_mapping[topic]}.csv'), index=False)

    def loadData(self):
        """
        Loop through all experiments of one series and load data:
        x: [x, y, theta], tx: time stamp of x, u: [u1, ..., u6], tu: time stamp of u
        """

        # loop through all sub folders of the defined experiment series
        for root, _, files in os.walk(os.path.join('experiments', self.series_name)):

            # loop through all files in subfolder
            for f in files:
                # skip if it is not a .csv file
                if f.find('.csv') == -1:
                    continue

                # add experiment either to optitrack of control data
                if f.find('optitrack') != -1:
                    
                    tx_sec = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[3], dtype=float)
                    tx_nano_sec = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[2], dtype=float)
                    self._tx[root] = tx_sec + tx_nano_sec * 1e-9
                    
                    self._x[root] = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[11, 12, 6], dtype=float)
                elif f.find('control') != -1:
                    tu_sec = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[3], dtype=float)
                    tu_nano_sec = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[2], dtype=float)
                    self._tu[root] = tu_sec + tu_nano_sec * 1e-9
                    
                    self._u[root] = np.genfromtxt(os.path.join(root, f), delimiter=',', skip_header=1, usecols=[4, 5, 6, 7, 8, 9], dtype=float)

        keys = list(self._x.keys())
        keys.extend(list(self._u.keys()))
        self.exps = np.unique(keys)

        self._crop_time()
        self._normalize_time()
    
    def _crop_time(self):
        """¨
        Crop data to be in time interval
        """

        for exp in self._tx:

            start_time = 0
            end_time = np.inf

            config_path = os.path.join(exp, 'config.json')
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    start_time = config.get('start_time', 0)
                    end_time = config.get('end_time', np.inf)

            if exp in self._tx:
                mask = np.logical_and(start_time <= self._tx[exp], self._tx[exp] <= end_time)
                self._tx[exp] = self._tx[exp][mask]
                self._x[exp] = self._x[exp][mask, :]

            if exp in self._tu:
                mask = np.logical_and(start_time <= self._tu[exp], self._tu[exp] <= end_time)
                self._tu[exp] = self._tu[exp][mask]
                self._u[exp] = self._u[exp][mask, :]

    def _normalize_time(self):
        """¨
        Normalizes time to start at 0
        """
        for exp in self._tx:
            # get starting time stamp
            if exp in self._tx:
                tx_min = self._tx[exp][0] 
            else:
                tx_min = np.Inf

            if exp in self._tu:
                tu_min = self._tu[exp][0] 
            else:
                tu_min = np.Inf
            
            start_stamp = np.min([tx_min, tu_min])

            # convert tx, tu and tf from stamp to seconds
            if exp in self._tx: self._tx[exp] = self._tx[exp] - start_stamp
            if exp in self._tu: self._tu[exp] = self._tu[exp] - start_stamp
