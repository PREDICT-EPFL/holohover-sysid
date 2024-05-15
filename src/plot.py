# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.params import Params
from src.holohover_model import HolohoverModel
from src.holohover_dataset import HolohoverDataset
from src.learn import Learn

class Plot():
    def __init__(self, params: Params, model: HolohoverModel, learn: Learn, device) -> None:
        self.params = params
        self.model = model
        self.learn = learn
        self.device = device

        self.frequency = 240
        self.plot_range = [int(5*self.frequency), int(8*self.frequency)]
        # self.plot_range = [int(10*self.frequency), int(13*self.frequency)]

    def greyModel(self):
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize =(20, 10))

        self._modelLoss(axs[0,0], losses_tr=self.learn.metrics['losses_tr'], losses_te=self.learn.metrics['losses_te'])
        self._modelError((axs[1,0], axs[2,0]), abs_error=self.learn.metrics['abs_error'], rms_error=self.learn.metrics['rms_error'])
        self._modelData((axs[0,1],axs[1,1],axs[2,1],axs[0,2],axs[1,2],axs[2,2],axs[0,3],axs[1,3],axs[2,3]), model_name='model (trained)', plot_white=True)

        plt.savefig(os.path.join(self.params.dir_path, 'learn_model.pdf'))

    def paramsSig2Thrust(self):

        U = torch.linspace(0, 1, steps=100, device=self.device).reshape(100,1).repeat(1,6)
        with torch.no_grad():
            thrust_learned = self.model.signal2thrust(U)
        thrust_learned = thrust_learned.detach().cpu().numpy()

        white_model = HolohoverModel(params=self.params, device=self.device)
        with torch.no_grad():
            thrust_init = white_model.signal2thrust(U)
        thrust_init = thrust_init.detach().cpu().numpy()

        U = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize =(6, 5))
        colors = ['royalblue', 'orange', 'green', 'red', 'darkorchid', 'brown']
        for i in range(thrust_learned.shape[1]):           
            ax.plot(U, thrust_learned[:,i], color=colors[i], label=f'Motor {i+1}')
            ax.plot(U, thrust_init[:,i], '--', color=colors[i])
        ax.set_ylabel('Thrust [N]')
        ax.set_xlabel('Signal')
        ax.legend()

        plt.savefig(os.path.join(self.params.dir_path, 'signal2thrust_params.pdf')) 

    def paramsVec(self):
        pos_learned = self.model.motors_pos.detach().cpu().numpy()
        vec_learned = self.model.motors_vec.detach().cpu().numpy()

        white_model = HolohoverModel(params=self.params, device='cpu')
        pos_init = white_model.motors_pos.detach().cpu().numpy()
        vec_init = white_model.motors_vec.detach().cpu().numpy()

        com_learned = self.model.center_of_mass.detach().cpu().numpy()
        com_init = white_model.center_of_mass.detach().cpu().numpy()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize =(6, 5))
        circle = plt.Circle((0, 0), 0.06, color='black', alpha=0.2, label='Holohover')
        ax.add_patch(circle)

        scaling_vec = 0.008
        vec_learned = vec_learned*scaling_vec
        vec_init = vec_init*scaling_vec
        for i in range(pos_init.shape[0]): 
            arrow =  mpatches.FancyArrow(pos_init[i,0], pos_init[i,1], vec_init[i,0], vec_init[i,1], 
                                        color='blue', length_includes_head=True)
            ax.add_patch(arrow)
            arrow =  mpatches.FancyArrow(pos_learned[i,0], pos_learned[i,1], vec_learned[i,0], vec_learned[i,1], 
                                        color='green', length_includes_head=True)
            ax.add_patch(arrow)   
        
        ax.scatter(com_init[0], com_init[1], color='blue', label=f'Before learning')
        ax.scatter(com_learned[0], com_learned[1], color='green', label=f'After learning')
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.set_ylabel('position [m]')
        ax.set_xlabel('position [m]')
        ax.set_xlim([-0.1,0.12])
        ax.set_ylim([-0.1,0.12])
        ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        plt.savefig(os.path.join(self.params.dir_path, 'pos_vec_params.pdf'))

    def dataHistogram(self):
        dataset = HolohoverDataset(self.params['data']['experiment'], self.plot_range[-1], with_dX=True)
        X = None
        U = None
        dX = None
        for exp in dataset.data['exps']:
            if X is None:
                X = dataset.data['X'][exp]
                U = dataset.data['U'][exp]
                dX = dataset.data['dX'][exp]
            else:
                X = torch.cat((X, dataset.data['X'][exp]), dim=0)
                U = torch.cat((U, dataset.data['U'][exp]), dim=0)
                dX = torch.cat((dX, dataset.data['dX'][exp]), dim=0)
        X = X.detach().numpy()
        U = U.detach().numpy()
        dX = dX.detach().numpy()

        U = np.concatenate((np.maximum(U[:,0], U[:,1]), np.maximum(U[:,2], U[:,3]), np.maximum(U[:,4], U[:,5])))

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(8, 7))
        fig.suptitle(f'Total nb. samples: {X.shape[0]}')
        axs[0,0].hist(U, bins=100, label='signal')
        axs[1,0].hist(dX[:,3], bins=100, label='dd(x)')
        axs[1,1].hist(dX[:,4], bins=100, label='dd(y)')
        axs[0,1].hist(dX[:,5], bins=100, label='dd(theta)')
        axs[0,0].set_xlabel('signal')
        axs[1,0].set_xlabel('[m/s^2]')
        axs[1,1].set_xlabel('[m/s^2]')
        axs[0,1].set_xlabel('[rad/s^2]')
        axs[0,0].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[0,1].legend()
        plt.savefig(os.path.join(self.params.dir_path, 'histogram.pdf'))

    def _modelLoss(self, axis, losses_tr, losses_te, log_scale=False):     
        axis.set_title(f'Loss')
        axis.set_ylabel('loss')
        axis.plot(losses_te, label='testing', color='purple')
        axis.plot(losses_tr, '--', label='training', color='purple')
        axis.set_yscale('log')
        axis.legend()

        if log_scale:
            axis.set_yscale('log')

    def _modelData(self, axs, model_name, plot_white=False):
        dataset = HolohoverDataset(self.params['data']['experiment'], self.plot_range[-1], with_dX=True)
        T, X, U, dX_real = dataset[0]

        T = T.to(self.device)
        X = X.to(self.device)
        U = U.to(self.device)

        T = T[self.plot_range[0]:self.plot_range[1]]
        X = X[self.plot_range[0]:self.plot_range[1],:]
        U = U[self.plot_range[0]:self.plot_range[1],:]
        dX_real = dX_real[self.plot_range[0]:self.plot_range[1],:]

        with torch.no_grad():
            dX_model = self.learn.dX_model(T, X, U)
            X_predict_stack = None
            i = 0
            while i + self.params['learning_params']['encoder_length'] + self.params['learning_params']['prediction_length'] < T.shape[0]:
                _, X_predict = self.learn.predict_system(T[i:].unsqueeze(0), X[i:,:].unsqueeze(0), U[i:,:].unsqueeze(0), self.params['learning_params']['encoder_length'], self.params['learning_params']['prediction_length'])
                X_predict = X_predict.squeeze(0)
                if X_predict_stack is None:
                    X_predict_stack = X_predict
                else:
                    X_predict_stack = torch.cat((X_predict_stack, X_predict), dim=0)
                i += self.params['learning_params']['prediction_length']

            if plot_white:
                params_white = Params()
                white_model = HolohoverModel(params=params_white, device=self.device)
                learn_white = Learn(params=self.params, dataset=dataset, model=white_model, device=self.device)
                dX_white = learn_white.dX_model(T, X, U)

        T_prediction = T[self.params['learning_params']['encoder_length']:self.params['learning_params']['encoder_length']+i]

        T = T.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        U = U.detach().cpu().numpy()
        T_prediction = T_prediction.detach().cpu().numpy()
        dX_model = dX_model.detach().cpu().numpy()
        X_predict_stack = X_predict_stack.detach().cpu().numpy()
        dX_white = dX_white.detach().cpu().numpy()

        axs[0].set_title(f'x')
        axs[0].set_ylabel('[m]')
        axs[0].plot(T, X[:,0], label='real', color='black')
        if plot_white:
            axs[0].plot(T_prediction, X_predict_stack[:,0], '--', label=model_name, color='cyan')

        axs[0].legend()

        axs[1].set_title(f'y')
        axs[1].set_ylabel('[m]')
        axs[1].plot(T, X[:,1], label='real', color='black')
        if plot_white:
            axs[1].plot(T_prediction, X_predict_stack[:,1], '--', label=model_name, color='cyan')
        axs[1].legend()

        axs[2].set_title(f'theta')
        axs[2].set_xlabel('time [s]')
        axs[2].set_ylabel('[rad]')
        axs[2].plot(T, X[:,2], label='real', color='black')
        if plot_white:
            axs[2].plot(T_prediction, X_predict_stack[:,2], '--', label=model_name, color='cyan')
        axs[2].legend()


        axs[3].set_title(f'd(x)')
        axs[3].set_ylabel('[m/s]')
        axs[3].plot(T, X[:,3], label='real', color='black')
        if plot_white:
            axs[3].plot(T_prediction, X_predict_stack[:,3], '--', label=model_name, color='cyan')

        axs[3].legend()

        axs[4].set_title(f'd(y)')
        axs[4].set_ylabel('[m/s]')
        axs[4].plot(T, X[:,4], label='real', color='black')
        if plot_white:
            axs[4].plot(T_prediction, X_predict_stack[:,4], '--', label=model_name, color='cyan')
        axs[4].legend()

        axs[5].set_title(f'd(theta)')
        axs[5].set_xlabel('time [s]')
        axs[5].set_ylabel('[rad/s]')
        axs[5].plot(T, X[:,5], label='real', color='black')
        if plot_white:
            axs[5].plot(T_prediction, X_predict_stack[:,5], '--', label=model_name, color='cyan')
        axs[5].legend()


        axs[6].set_title(f'dd(x)')
        axs[6].set_ylabel('[m/s^2]')
        axs[6].plot(T, dX_real[:,3], label='real', color='black')
        if plot_white:
            axs[6].plot(T, dX_white[:,3], label='model (untrained)', color='blue')
            axs[6].plot(T, dX_model[:,3], '--', label=model_name, color='cyan')
            # axs[6].plot(T, dX_real[:, 3] - dX_model[:, 3], '--', label=model_name + 'dif.', color='orange')

        axs[6].legend()

        axs[7].set_title(f'dd(y)')
        axs[7].set_ylabel('[m/s^2]')
        axs[7].plot(T, dX_real[:,4], label='real', color='black')
        if plot_white:
            axs[7].plot(T, dX_white[:,4], label='model (untrained)', color='blue')
            axs[7].plot(T, dX_model[:,4], '--', label=model_name, color='cyan')
            # axs[7].plot(T, dX_real[:, 4] - dX_model[:, 4], '--', label=model_name + 'dif.', color='orange')
        axs[7].legend()

        axs[8].set_title(f'dd(theta)')
        axs[8].set_xlabel('time [s]')
        axs[8].set_ylabel('[rad/s^2]')
        axs[8].plot(T, dX_real[:,5], label='real', color='black')
        if plot_white:
            axs[8].plot(T, dX_white[:,5], label='model (untrained)', color='blue')
            axs[8].plot(T, dX_model[:,5], '--', label=model_name, color='cyan')
            # axs[8].plot(T, dX_real[:, 5] - dX_model[:, 5], '--', label=model_name + 'dif.', color='orange')
        axs[8].legend()

    def _modelError(self, axs, abs_error, rms_error):
        abs_error = np.array(abs_error)
        rms_error = np.array(rms_error)

        axs[0].set_title(f'Error')
        axs[0].set_ylabel('[m/s^2]')
        axs[0].plot(abs_error[:,3], label='abs dd(x)', color='red')
        axs[0].plot(rms_error[:,3], '--', label='rms dd(x)', color='red')
        axs[0].plot(abs_error[:,4], label='abs dd(y)', color='orange')
        axs[0].plot(rms_error[:,4], '--', label='rms dd(y)', color='orange')
        axs[0].legend()

        axs[1].set_title(f'Error')
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('[rad/s^2]')
        axs[1].plot(abs_error[:,5], label=' abs dd(theta)', color='gold')
        axs[1].plot(rms_error[:,5], '--', label='rms dd(theta)', color='gold')
        axs[1].legend()
