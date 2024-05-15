# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import os

from src.data import Data

class PreprocessPlot():
    def __init__(self, data: Data, show_plots: bool, save_plots: bool) -> None:
        self.data = data
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.xlim = [5, 8]

    def _save_plot(self, exp, name):
        path = os.path.join(exp, 'plots', name)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)

    def cropData(self, plot_lw_x_idx, plot_up_x_idx):
        """
        Plot removed data in the beginning and the end
        """
        x, tx = self.data.get(names=['x', 'tx'])

        for exp in self.data.exps:
            plt.plot(tx[exp], x[exp][:,2], )
            plt.vlines(tx[exp][plot_lw_x_idx[exp]], ymin=-3.3, ymax=3.3, colors='r')
            plt.vlines(tx[exp][plot_up_x_idx[exp]], ymin=-3.3, ymax=3.3, colors='r')
            plt.title(f'Theta of exp. {exp[30:]}')
            plt.xlabel('time [s]')
            plt.ylabel('angle [rad]')

            if self.show_plots:
                plt.show()
            if self.save_plots:
                self._save_plot(exp, 'crop_data.pdf')

    def interpolateU(self, u_inter):
        """
        Plot polynomial interpolation of control input to match with x
        Args:
            u_inter: interpolation of u, dict
        """
        tx, u, tu = self.data.get(names=['tx', 'u', 'tu'])
        u_inter = u_inter.copy()

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(8, 8))             
            for i in range(u[exp].shape[1]):
                k = int(i/len(axs[0]))
                l = i % len(axs[0])

                axs[k,l].plot(tu[exp], u[exp][:,i], color='b', label=f'u({i+1})')
                axs[k,l].plot(tx[exp], u_inter[exp][:,i], '--', color='r', label=f'u({i+1}) inter.')
                axs[k,l].legend()
                axs[k,l].set_xlim(self.xlim)

            axs[0,0].set_ylabel('signal')
            axs[1,0].set_ylabel('signal')
            axs[2,0].set_ylabel('signal')
            axs[2,0].set_xlabel('time [s]')
            axs[2,1].set_xlabel('time [s]')

            if self.show_plots:
                plt.show()
            if self.save_plots:
                self._save_plot(exp, 'interpolate_u.pdf')

    def firstOrderMotorSpeed(self, u, motor_speed):
        """
        Plot first order model approximation of U
        Args:
            u: u, dict
            motor_speed: first order model approximation of u (motor speed), dict
        """
        tx, u = self.data.get(names=['tx', 'u'])

        for exp in self.data.exps:

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(8, 8))             
            for i in range(u[exp].shape[1]):
                k = int(i/len(axs[0]))
                l = i % len(axs[0])

                axs[k,l].plot(tx[exp], u[exp][:,i], color='b', label=f'u({i+1})')
                axs[k,l].plot(tx[exp], motor_speed[exp][:,i], '--', color='r', label=f'motor speed')
                axs[k,l].legend()
                axs[k,l].set_xlim(self.xlim)
 
            axs[0,0].set_ylabel('thrust')
            axs[1,0].set_ylabel('thrust')
            axs[2,0].set_ylabel('thrust')
            axs[2,0].set_xlabel('time [s]')
            axs[2,1].set_xlabel('time [s]')
            
            if self.show_plots:
                plt.show()
            if self.save_plots:
                self._save_plot(exp, 'first_order_motor_speed.pdf')

    def diffPosition(self):
        tx, x, dx, ddx = self.data.get(names=['tx', 'x', 'dx', 'ddx'])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=x[exp].shape[1], ncols=2, figsize =(8, 8))             
            for i, ax in enumerate(axs[:,0]):
                if i==0: label = 'x'
                elif i==1: label = 'y'
                elif i==2: label = 'theta'

                ax.plot(tx[exp], x[exp][:,i], color='b', label='pos('+label+')')
                ax.plot(tx[exp], dx[exp][:,i], color='g', label='vel('+label+')')
                ax.plot(tx[exp], ddx[exp][:,i], color='r', label='acc('+label+')')
                ax.legend(ncol=2)
                ax.set_xlim(self.xlim)

            # integrate dx and calc. error
            d_error = self._integrate(dx[exp], tx[exp], x[exp][0,:])
            d_error = d_error - x[exp]

            # double integrate imu and calc. error
            dd_error = self._integrate(ddx[exp], tx[exp], dx[exp][0,:])
            dd_error = self._integrate(dd_error, tx[exp], x[exp][0,:])
            dd_error = dd_error - x[exp]

            for i, ax in enumerate(axs[:,1]):
                if i==0: label = 'x'
                elif i==1: label = 'y'
                elif i==2: label = 'theta'

                ax.plot(tx[exp], d_error[:,i], color='g', label='dx('+label+') error')
                ax.plot(tx[exp], dd_error[:,i], color='r', label='ddx('+label+') error')
                ax.legend()

            axs[2,0].set_xlabel('time [s]')
            axs[2,1].set_xlabel('time [s]')          

            if self.show_plots:
                plt.show()
            if self.save_plots:
                self._save_plot(exp, 'diff_position.pdf')

    def uShift(self, total_shift, ddx_u, ddx_u_unshifted, ddx_unshifted):
        """
        Args:
            total_shift: nb. data shifts
            ddx_u: white box estimation of ddx using u, dict
            ddx_u_unshifted: white box estimation of ddx using u before shifting data, dict
            ddx_unshifted: differentiated opti-track ddx before shifting, dict
        """
        tx, x, ddx = self.data.get(names=['tx', 'x', 'ddx'])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=x[exp].shape[1], ncols=1, figsize =(8, 11))   

            delay_s = total_shift * (tx[exp][-1] - tx[exp][0]) / tx[exp].shape[0]
            
            fig.suptitle(f'Avg. delay of ddx wrt. control: {np.round(delay_s*1000,1)}ms ({total_shift} shifts)')
            for i, ax in enumerate(axs):
                ax.plot(tx[exp], ddx_unshifted[exp][:ddx_u[exp].shape[0],i], '--', color='g', label='Optitrack (not shifted)')
                ax.plot(tx[exp], ddx[exp][:,i],  color='r', label='Optitrack')
                ax.plot(tx[exp], ddx_u_unshifted[exp][:ddx_u[exp].shape[0],i], '--', color='b', label='Model (not shifted)')
                ax.plot(tx[exp], ddx_u[exp][:,i], color='c', label='Model')
                ax.legend()
                error = (1/ddx_u[exp].shape[0]) * np.sum(np.abs(ddx_u[exp][:,i]-ddx[exp][:,i]))
                unshifted_error = (1/ddx_u[exp].shape[0]) * np.sum(np.abs(ddx_u[exp][:,i]-ddx_unshifted[exp][:ddx_u[exp].shape[0],i]))
                ax.set_title(f'Mean abs. error: {np.round(error,3)} (shifted), {np.round(unshifted_error,3)} (not shifted)')
                ax.set_xlim(self.xlim)


            axs[2].set_xlabel('time [s]')
            axs[0].set_ylabel('dd(x) [m/s^2]') 
            axs[1].set_ylabel('dd(y) [m/s^2]') 
            axs[2].set_ylabel('dd(theta) [rad/s^2]') 
            
            if self.show_plots:
                plt.show()
            if self.save_plots:
                self._save_plot(exp, 'u_shift.pdf')

    def _integrate(self, dX, tX, X0):
        """
        Integrate dX
        Args:
            dX: derivative of X, (N, D)
            tX: time in seconds of X, (N)
            X0: initial condition of X, (D)
        Returns:
            X: integration of dX (N, D)
        """
        dX = np.copy(dX)
        X = np.zeros(dX.shape)
        X[0,:] = X0
        for i in range(1,dX.shape[0]):
            X[i,:] = X[i-1,:] + dX[i,:]*(tX[i]-tX[i-1])
        
        return X