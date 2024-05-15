# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import time
import os
import torch
from torch.utils.data import random_split, DataLoader

from src.params import Params
from src.holohover_dataset import HolohoverDataset
from src.holohover_model import HolohoverModel

class Learn:
    def __init__(self, params: Params, dataset: HolohoverDataset, model: HolohoverModel, device):
        """
        Args:
            params: parameter instance
            dataset: dataset
            model: model class instance
            device: pytorch device
        """
        self.params = params
        self.dataset = dataset
        self.device = device
        self.model = model.to(self.device)

        # logging data
        self.metrics = { "losses_tr":[], "losses_te":[], "abs_error":[], "rms_error":[], "std_error":[] }

        model_params = [{'params':self.model.sig2thr_fcts[i].weight, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_signal2thrust'] } for i in range(len(self.model.sig2thr_fcts))]
        model_params.append({'params': self.model.configuration_matrix, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_configuration_matrix'] })
        model_params.append({'params': self.model.center_of_mass_param, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_center_of_mass'] })
        model_params.append({'params': self.model.mass, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_mass'] })
        model_params.append({'params': self.model.tau, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_tau'] })
        model_params.append({'params': self.model.inertia, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_inertia'] })
        model_params.append({'params': self.model.motor_angle_offset, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_motor_angle_offset'] })
        model_params.append({'params': self.model.motors_vec_param, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_motors_vec'] })
        model_params.append({'params': self.model.motors_pos_param, 'lr': params['learning_params']['learning_rate'] * params['model']['lr_mul_motors_pos'] })
        self.optimizer = torch.optim.Adam(model_params, lr=params['learning_params']['learning_rate'])

    def augment_system(self, T, X, U):

        speed_first_order = torch.zeros_like(U, device=self.device)
        speed_first_order[:, 0, :] = U[:, 0, :]
        for i in range(1, U.shape[1]):
            if self.model.tau == 0:
                speed_first_order[:, i, :] = U[:, i, :]
            else:
                dt = (T[:, i] - T[:, i - 1]).unsqueeze(-1)
                k1 = self.model.firstOrderMotorSpeedModel(speed_first_order[:, i - 1, :], U[:, i, :])
                k2 = self.model.firstOrderMotorSpeedModel(speed_first_order[:, i - 1, :] + dt * k1 / 2, U[:, i, :])
                k3 = self.model.firstOrderMotorSpeedModel(speed_first_order[:, i - 1, :] + dt * k2 / 2, U[:, i, :])
                k4 = self.model.firstOrderMotorSpeedModel(speed_first_order[:, i - 1, :] + dt * k3, U[:, i, :])
                speed_first_order[:, i, :] = speed_first_order[:, i - 1, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        X_augment = torch.cat((X, speed_first_order), dim=2)

        return X_augment
    
    def dX_model(self, T, X, U):

        X_augment = self.augment_system(T.unsqueeze(0), X.unsqueeze(0), U.unsqueeze(0)).squeeze(0)
        dX = self.model(X_augment, U)
        return dX

    def predict_system(self, T, X, U, encoder_length, prediction_length):

        X_augment = self.augment_system(T, X, U)

        X_predict = torch.zeros((X.shape[0], prediction_length, X_augment.shape[2]), dtype=torch.float, device=self.device)
        X_current = X_augment[:, encoder_length - 1, :]
        for i in range(0, prediction_length):
            i_full = i + encoder_length
            dt = (T[:, i_full] - T[:, i_full - 1]).unsqueeze(-1)
            k1 = self.model(X_current, U[:, i_full - 1])
            k2 = self.model(X_current + dt * k1 / 2, U[:, i_full - 1])
            k3 = self.model(X_current + dt * k2 / 2, U[:, i_full - 1])
            k4 = self.model(X_current + dt * k3, U[:, i_full - 1])
            X_current = X_current + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X_predict[:, i, :] = X_current

            if self.model.tau == 0:
                X_predict[:, i, 6:] = U[:, i_full - 1]

        return X_augment[:, encoder_length:, :], X_predict

    def lossFunction(self, X_real, X_predict):

        # scaling = torch.ones(12, dtype=torch.float, device=self.device)
        # scaling[5] = 0.1
        # loss = torch.mean(torch.square(X_real - X_predict) * scaling.unsqueeze(0).unsqueeze(0))

        scaling = torch.ones(3, dtype=torch.float, device=self.device)
        scaling[2] = 0.1
        loss = torch.mean(torch.square((X_real[:, :, 0:3] - X_predict[:, :, 0:3]) * scaling))

        # regularizer to ensure signal2thrust is monotonically increasing
        eval_points = torch.linspace(0, 1, steps=20, device=self.device)
        if self.params['model']['thrust_poly_order'] == 3:
            for lin_fct in self.model.sig2thr_fcts:
                a, b, c = lin_fct.weight[0, 0], lin_fct.weight[0, 1], lin_fct.weight[0, 2]
                signal2thrust_deriv = a + 2 * b * eval_points + 3 * c * eval_points
                reg_weight = 1e3
                loss += torch.sum(torch.nn.functional.relu(-a)) * reg_weight
                loss += torch.sum(torch.nn.functional.relu(-signal2thrust_deriv)) * reg_weight
                if c > 0 and (-b / (3 * c) >= 0 and -b / (3 * c) <= 1):
                    loss += torch.sum(torch.nn.functional.relu(b ** 2 / (3 * c) - a)) * reg_weight

        return loss

    def evaluate(self, dataloader):

        X_augment_stacked = None
        X_predict_stacked = None
        with torch.no_grad():
            for T, X, U in dataloader:
                T = T.to(self.device)
                X = X.to(self.device)
                U = U.to(self.device)

                X_augment, X_predict = self.predict_system(T, X, U, self.params['learning_params']['encoder_length'], self.params['learning_params']['prediction_length'])

                if X_augment_stacked is None:
                    X_augment_stacked = X_augment
                    X_predict_stacked = X_predict
                else:
                    X_augment_stacked = torch.cat((X_augment_stacked, X_augment), dim=0)
                    X_predict_stacked = torch.cat((X_predict_stacked, X_predict), dim=0)

        loss = self.lossFunction(X_augment_stacked, X_predict_stacked)
        abs_error = torch.mean(torch.abs(X_augment_stacked - X_predict_stacked), dim=(0, 1))
        rms_error = torch.sqrt(torch.mean(torch.square(X_augment_stacked - X_predict_stacked), dim=(0, 1)))
        std_error = torch.std(X_augment_stacked - X_predict_stacked, dim=(0, 1))

        self.metrics["losses_te"].append(loss.detach().cpu().float())
        self.metrics["abs_error"].append(abs_error.detach().cpu().numpy())
        self.metrics["rms_error"].append(rms_error.detach().cpu().numpy())
        self.metrics["std_error"].append(std_error.detach().cpu().numpy())

    def printMetrics(self, epoch, elapse_time):
        """
        Prints testing and training loss as well as abs. and RMS error
        Args:
            epoch: current epoch, int
            elapse_time: elapse time of epoch, float
        """
        loss_te = self.metrics["losses_te"][-1]
        loss_tr = self.metrics["losses_tr"][-1]
        abs_error = self.metrics["abs_error"][-1][3:6]
        rms_error = self.metrics["rms_error"][-1][3:6]
        std_error = self.metrics["std_error"][-1][3:6]
        print(f"Epoch {epoch}: \telapse time = {np.round(elapse_time,3)}, \ttesting loss = {loss_te}, \ttraining loss = {loss_tr}")
        print(f"Epoch {epoch}: \tabs error = {np.round(abs_error,4)}, \trms error = {np.round(rms_error,4)}, \tstd error = {np.round(std_error,4)}")

    def optimize(self):
        """
        Main optimization loop inclusive data loading, training and evaluating
        """
        learning_len = math.floor((1 - self.params['learning_params']['testing_share']) * len(self.dataset))
        testing_len = len(self.dataset) - learning_len
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(self.dataset, [learning_len, testing_len], generator)
        train_dataloader = DataLoader(train_dataset, batch_size=self.params['learning_params']['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.params['learning_params']['batch_size'], shuffle=True)

        self.metrics = { "losses_tr":[], "losses_te":[], "abs_error":[], "rms_error":[], "std_error":[] }
        self.evaluate(test_dataloader)
        self.metrics["losses_tr"].append(self.metrics["losses_te"][0])
        self.printMetrics(epoch=0, elapse_time=0.0)

        for j in range(self.params['learning_params']['nb_epochs']):
            start_time = time.time()

            loss_tr = []
            for T, X, U in train_dataloader:
                T = T.to(self.device)
                X = X.to(self.device)
                U = U.to(self.device)

                def closure():
                    self.optimizer.zero_grad()
                    X_augment, X_predict = self.predict_system(T, X, U, self.params['learning_params']['encoder_length'], self.params['learning_params']['prediction_length'])
                    loss = self.lossFunction(X_augment, X_predict)
                    loss_tr.append(loss.detach().cpu().float())
                    loss.backward()
                    return loss
                
                self.optimizer.step(closure=closure)

            self.metrics["losses_tr"].append(np.mean(loss_tr))
            self.evaluate(test_dataloader)
            self.printMetrics(epoch=j+1, elapse_time=time.time()-start_time)

    def saveModel(self):
        """
        Save model parameters
        """
        torch.save(self.model.state_dict(), os.path.join(self.params.dir_path, 'model.pt'))
