# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
import os
import shutil
import torch

from src.params import Params
from src.holohover_dataset import HolohoverDataset
from src.holohover_model import HolohoverModel
from src.learn import Learn
from src.plot import Plot


def main():
    # pytorch device and random seed
    # if torch.cuda.is_available():  
    #     dev = 'cuda:0' 
    # else:  
    #     dev = 'cpu'

    dev = 'cpu'
    torch.set_num_threads(4)

    device = torch.device(dev)
    torch.manual_seed(0)

    # load parameters
    params = Params()

    # create directory
    t = datetime.now()
    dir_name = t.strftime('%Y_%m_%d-%H_%M_%S')
    params.dir_path = os.path.join('models', dir_name)
    if os.path.exists(params.dir_path):
        shutil.rmtree(params.dir_path)
    os.mkdir(params.dir_path)

    # init. model
    model = HolohoverModel(params=params, device=device)
    params.set_model_params(model, 'model_params_init')

    # init. base learner
    dataset = HolohoverDataset(params['data']['experiment'], params['learning_params']['encoder_length'] + params['learning_params']['prediction_length'])
    ld = Learn(params=params, dataset=dataset, model=model, device=device)

    # learn dynamics
    ld.optimize()

    # plot results
    plot = Plot(params=params, model=model, learn=ld, device=device)
    plot.greyModel()
    plot.paramsSig2Thrust()
    plot.paramsVec()
    plot.dataHistogram()

    # save model and parameters
    ld.saveModel()
    params.set_model_params(model, 'model_params')
    params.save()


if __name__ == '__main__':
    main()
