# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from src.params import Params
from src.utils import poly_expand_u

class HolohoverModel(nn.Module):
    def __init__(self, params: Params, device):
        """
        Args:
            params: parameter class instance
            dev: pytorch device
        """
        super(HolohoverModel, self).__init__()

        self.params = params
        self.device = device 

        # system parameters
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of control input u   
        self.S = 3 # dim. of space

        # Center of mass
        self.init_center_of_mass = Parameter(torch.tensor(params['default_model_params']['center_of_mass'], device=device), requires_grad=False)
        self.center_of_mass_param = Parameter(self.init_center_of_mass[0:2].clone(), requires_grad=params['model']['learn_center_of_mass'])
        # mass
        self.mass = Parameter(torch.tensor(params['default_model_params']['mass'], device=device), requires_grad=params['model']['learn_mass'])
        # Inertia around z axis
        self.inertia = Parameter(torch.tensor(params['default_model_params']['inertia'], device=device), requires_grad=params['model']['learn_inertia'])

        self.tau = Parameter(torch.tensor(params['default_model_params']['tau'], device=device), requires_grad=params['model']['learn_tau'])

        # angle offset
        self.motor_angle_offset = Parameter(torch.tensor(params['default_model_params']['motor_angle_offset'], device=device), requires_grad=params['model']['learn_motor_angle_offset'])

        # motor positions
        self.init_motors_pos = Parameter(self.defaultMotorPos().clone(), requires_grad=False)
        self.motors_pos_param = Parameter(self.init_motors_pos[:, 0:2].clone(), requires_grad=params['model']['learn_motors_pos'])

        # unit vectors of thrust from motors
        self.init_motors_vec = Parameter(self.defaultMotorVec().clone(), requires_grad=False)
        self.motors_vec_param = Parameter(self.init_motors_vec[:, 0:2].clone(), requires_grad=params['model']['learn_motors_vec'])

        # calculate initial configuration matrix from parameters
        def combined_thrust2body_acc(thrust):
            linear_acc, angular_acc = self.thrust2body_acc(thrust.unsqueeze(0))
            return (linear_acc + angular_acc).squeeze(0)
        rand_thrust = torch.rand(self.M, device=device)
        thrust2acc_jac = torch.autograd.functional.jacobian(combined_thrust2body_acc, rand_thrust)

        # configuration matrix
        self.configuration_matrix = Parameter(thrust2acc_jac.clone(), requires_grad=params['model']['learn_configuration_matrix'])

        # if thrust_poly_order < 3 or thrust_poly_order > 3, then signal2thrust must be cropped or extended resp. 
        init_signal2thrust = torch.tensor(params['default_model_params']['signal2thrust'], device=device)
        if params['model']['thrust_poly_order'] < 3: # desired polynomial expansion has smaller degree than measured coeff. -> ignore higher coeff.
            init_signal2thrust = init_signal2thrust[:,:params['model']['thrust_poly_order']]
        elif params['model']['thrust_poly_order'] > 3: # desired polynomial expansion has larger degree than measured coeff. -> add coeff. = zero
            padding = torch.zeros(self.M, params['model']['thrust_poly_order'] - 3, device=device)
            init_signal2thrust = torch.concat((init_signal2thrust, padding), axis=1)

        # signal2thrust mapping
        input_size = params['model']['thrust_poly_order']
        output_size = 1
        self.sig2thr_fcts = nn.ModuleList([nn.Linear(input_size, output_size, bias=False, dtype=torch.float, device=device) for _ in range(self.M)])
        for i, lin_fct in enumerate(self.sig2thr_fcts):
            lin_fct.weight = Parameter(init_signal2thrust[i,:].clone().reshape(1, params['model']['thrust_poly_order']), requires_grad=params['model']['learn_signal2thrust'])

        if params['model']['load_model']:
            self.load_state_dict(torch.load(params['model']['model_path']))

    @property
    def center_of_mass(self):
        return torch.cat((self.center_of_mass_param, self.init_center_of_mass[2:]))
    
    @property
    def motors_pos(self):
        if self.params['model']['learn_motors_pos']:
            return torch.cat((self.motors_pos_param, self.init_motors_pos[:, 2:]), dim=1)
        else:
            return self.defaultMotorPos()
    
    @property
    def motors_vec(self):
        if self.params['model']['learn_motors_vec']:
            return torch.cat((self.motors_vec_param, self.init_motors_vec[:, 2:]), dim=1)
        else:
            return self.defaultMotorVec()

    def rotMatrix(self, theta):
        """
        Calc. 3D rotational matrix for batch
        Args:
            theta: rotation aroung z-axis in world frame, tensor (N)
        Returns:
            rot_mat: rotational matrix, tensor (N,S,S)
        """
        rot_mat = torch.zeros(theta.shape[0], self.S, self.S, device=self.device)
        cos = torch.cos(theta) # (N)
        sin = torch.sin(theta) # (N)
        rot_mat[:,0,0] = cos
        rot_mat[:,1,1] = cos
        rot_mat[:,0,1] = -sin
        rot_mat[:,1,0] = sin
        rot_mat[:,2,2] = torch.ones(theta.shape[0])
        return rot_mat

    def defaultMotorPos(self):
        """
        Initiate motors position
        Returns:
            motors_pos: position of motors in robot frame, tensor (M,S)
        """ 
        motors_pos = torch.zeros(self.M, self.S, dtype=torch.float, device=self.device)

        for j in np.arange(0, self.M, step=2):
            angle_motor_pair = self.motor_angle_offset  + j * np.pi / 3
            angle_first_motor = angle_motor_pair - self.params['default_model_params']['motor_angel_delta']
            angle_second_motor = angle_motor_pair + self.params['default_model_params']['motor_angel_delta']

            motors_pos[j,:] = self.params['default_model_params']['motor_distance'] * torch.stack([torch.cos(angle_first_motor), torch.sin(angle_first_motor), torch.tensor(0.0, device=self.device)])
            motors_pos[j+1,:] = self.params['default_model_params']['motor_distance'] * torch.stack([torch.cos(angle_second_motor), torch.sin(angle_second_motor), torch.tensor(0.0, device=self.device)])

        return motors_pos

    def defaultMotorVec(self):
        """
        Initiate motors vector
        Returns:
            motors_vec: unit vector pointing in direction of thrust from each motor, tensor (M,S)
        """ 
        motors_vec = torch.zeros(self.M, self.S, device=self.device)

        for j in np.arange(0, self.M, step=2):
            angle_motor_pair = self.motor_angle_offset  + j*np.pi/3
            motors_vec[j,:] = torch.stack([-torch.sin(angle_motor_pair), torch.cos(angle_motor_pair), torch.tensor(0.0, device=self.device)])
            motors_vec[j+1,:] = torch.stack([torch.sin(angle_motor_pair), -torch.cos(angle_motor_pair), torch.tensor(0.0, device=self.device)])

        return motors_vec

    def forward(self, X, U):
        """
        Forward pass through main model
        Args:
            X: state input batch (N, D)
            U: control input batch (N, M)
        Returns:
            dX_X: state derivative (N, D)
        """
        acc = self.signal2acc(X, U)
        d_speed = self.firstOrderMotorSpeedModel(X[:, 6:], U)
        dX_X = torch.concat((X[:,3:6], acc, d_speed), axis=1)

        return dX_X


    def signal2acc(self, X, U):
        """
        Calc acceleration with current state and control input
        Args:
            X: state input batch (N, D)
            U: control input batch (N, M)
        Returns:
            acc: acceleration (N, S)
        """

        if self.tau == 0:
            speed = U
        else:
            speed = X[:, 6:] # (N, M)
        thrust = self.signal2thrust(speed)
        acc = self.thrust2acc(thrust, X) # (N, S)
        return acc

    def signal2thrust(self, U):
        """
        Motor signal to motor thrust mapping
        Args:
            U: motor signals batch, tensor (N, M)
        Returns:
            thrust: motor thrust, tensor (N, M)
        """
        U_poly = poly_expand_u(U, self.params['model']['thrust_poly_order'])

        assert U_poly.shape[-1] % self.M == 0
        deg = int(U_poly.shape[-1] / self.M) # degree of polynomial expansion

        thrust = torch.zeros((*U_poly.shape[0:-1], self.M), dtype=torch.float, device=self.device)
        for i, lin_fct in enumerate(self.sig2thr_fcts):
            thrust[..., i] = lin_fct(U_poly[..., int(i*deg):int((i+1)*deg)]).squeeze(-1)

        return thrust

    def thrust2body_acc(self, thrust):
        # calc. thrust vector for each motor
        motor_vec_norm = nn.functional.normalize(self.motors_vec, dim=1)
        thrust_vec = torch.einsum('nm,ms->nms', thrust, motor_vec_norm) # (N, M, S)

        # calc. sum of forces in body and world frame
        Fb_sum = torch.sum(thrust_vec, dim=1) # (N, S)

        # calc. sum of moments in body frame in respect to center of mass
        com2motor_vec = self.motors_pos - self.center_of_mass # (M, S)
        com2motor_vec = com2motor_vec.reshape(1, self.M, self.S).tile(thrust.shape[0], 1, 1) # (N, M, S)
        Mb = torch.linalg.cross(com2motor_vec, thrust_vec, dim=2) # (N, M, S)
        Mb_sum = torch.sum(Mb, dim=1) # (N, S)

        # calc. acceleration, Fw_sum[0,:] = [Fx, Fy, Fz] and Mb[0,:] = [Mx, My, Mz]
        # holohover moves in a plane -> Fz = Mx = My = 0, also Mz_body = Mz_world
        linear_acc = Fb_sum / self.mass
        angular_acc = Mb_sum / self.inertia

        return linear_acc, angular_acc

    def thrust2acc(self, thrust, X):
        """       
        Args: 
            thrust: norm of thrust from each motor, tensor (N, M)
            X: current state [x, y, theta, dx, dy, dtheta], tensor (N, D)
        Returns:
            acc: acceleration of holohover [ddx, ddy, ddtheta], tensor (N, S)
        """
        rotation_b2w = self.rotMatrix(X[:,2]) # (N, S, S)

        if self.params['model']['learn_configuration_matrix']:
            # configuration_matrix @ batched thrust vector
            linear_angular_acc = torch.matmul(thrust, self.configuration_matrix.T) # (N, S)
            # rotate into world frame
            linear_angular_acc = torch.einsum('nps,ns->np', rotation_b2w, linear_angular_acc)
            # separate linear and angular acceleration
            linear_acc = torch.cat((linear_angular_acc[:, 0:2], torch.zeros((thrust.shape[0], 1), dtype=torch.float, device=self.device)), dim=1)
            angular_acc = torch.cat((torch.zeros((thrust.shape[0], 2), dtype=torch.float, device=self.device), linear_angular_acc[:, 2:]), dim=1)

        else:
            linear_acc, angular_acc = self.thrust2body_acc(thrust)
            # rotate into world frame
            linear_acc = torch.einsum('nps,ns->np', rotation_b2w, linear_acc) # (N, S)

        # correction terms for linear acceleration
        # see https://physics.stackexchange.com/questions/249804/why-is-the-center-of-mass-frame-always-used-in-rigid-body-dynamics
        center2com = self.center_of_mass
        center2com = center2com.tile(thrust.shape[0], 1) # (N,S)
        linear_acc += torch.linalg.cross(center2com, angular_acc, dim=1)
        angular_vel = torch.cat((torch.zeros((thrust.shape[0], 2), dtype=torch.float, device=self.device), X[:, 5:6]), dim=1)
        linear_acc += torch.linalg.cross(angular_vel, torch.linalg.cross(center2com, angular_vel, dim=1), dim=1)

        return linear_acc + angular_acc

    def firstOrderMotorSpeedModel(self, speed, speed_ref):
        return (speed_ref - speed) / self.tau
