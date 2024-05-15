# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import torch

from src.params import Params
from src.holohover_model import HolohoverModel
from src.data import Data
from src.preprocess import Preprocess
from src.preprocess_plot import PreprocessPlot

def main():
    # if torch.cuda.is_available():  
    #     dev = "cuda:0" 
    # else:  
    #     dev = "cpu"

    dev = 'cpu'
    torch.set_num_threads(4)

    device = torch.device(dev)

    params = Params()
    model = HolohoverModel(params=params, device=device)
    data = Data(params)
    
    if params['data']['convert_mcap']:
        print('Converting mcap to csv...')
        data.convert_mcap_to_csv()

    print('Loading data...')
    data.loadData()

    plot = PreprocessPlot(data=data, show_plots=False, save_plots=True)
    pp = Preprocess(data=data, plot=plot, model=model)

    print('Making angles continuous...')
    pp.continuous_angle()

    print('Cropping data...')
    pp.cropData(plot=True)

    print('Interpolating u...')
    pp.interpolateU(plot=True)

    print('Calculating first order motor dynamics...')
    pp.firstOrderMotorSpeed(plot=True)

    print('Numerically differentiating data...')
    pp.diffX(plot=True)

    print('Align data...')
    pp.alignData(plot=True)

    data.save()

if __name__ == "__main__":
    main()
