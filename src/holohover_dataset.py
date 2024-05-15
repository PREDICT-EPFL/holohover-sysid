# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import math
from torch.utils.data import Dataset

class HolohoverDataset(Dataset):
    def __init__(self, series, T, with_dX=False, allow_overlapping=True) -> None:
        """
        Args:
            series: experiment series to load
            T: window size of trajectories
            with_dX: also return derivatives
            allow_overlapping: allow overlapping of trajectories
        """
        super().__init__()

        self.T = T
        self.with_dX = with_dX
        self.data = torch.load(os.path.join('experiments', series, 'data.pt'))
        self.allow_overlapping = allow_overlapping
        
    def __len__(self):
        total_len = 0
        
        for exp in self.data['exps']:
            if self.allow_overlapping:
                total_len += len(self.data['T'][exp]) + 1 - self.T
            else:
                total_len += math.floor(len(self.data['T'][exp]) / self.T)

        return total_len
    
    def __getitem__(self, idx):
        idx_offset = 0
        for exp in self.data['exps']:
            local_idx = idx - idx_offset
            if self.allow_overlapping:
                len_exp = len(self.data['T'][exp]) + 1 - self.T
            else:
                len_exp = math.floor(len(self.data['T'][exp]) / self.T)
            if local_idx >= len_exp:
                idx_offset += len_exp
                continue

            if not self.allow_overlapping:
                local_idx *= self.T

            T = self.data['T'][exp][local_idx:(local_idx + self.T)].float()
            X = self.data['X'][exp][local_idx:(local_idx + self.T), :].float()
            U = self.data['U'][exp][local_idx:(local_idx + self.T), :].float()

            if self.with_dX:
                dX = self.data['dX'][exp][local_idx:(local_idx + self.T), :].float()
                return T, X, U, dX

            return T, X, U
