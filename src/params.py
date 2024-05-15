# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import jsbeautifier

class Params():
    def __init__(self, path='params.json') -> None:

        with open(path) as f:
            self.params = json.load(f)

        self.dir_path = None

    def __getitem__(self, key):
        return self.params[key]
    
    def __setitem__(self, key, value):
        self.params[key] = value

    def set_model_params(self, model, field='model_params'):

        sig2thr_list = []
        for lin_fct in model.sig2thr_fcts:
            sig2thr_list.append(list(lin_fct.weight.detach().cpu().numpy().flatten().astype(float)))

        self.params[field] = {}
        self.params[field]['center_of_mass'] = list(model.center_of_mass.detach().cpu().numpy().astype(float))
        self.params[field]['mass'] = model.mass.detach().cpu().numpy().astype(float).item()
        self.params[field]['inertia'] = model.inertia.detach().cpu().numpy().astype(float).item()
        self.params[field]['configuration_matrix'] = model.configuration_matrix.detach().cpu().numpy().tolist()
        self.params[field]['signal2thrust'] = sig2thr_list
        self.params[field]['motor_angle_offset'] = model.motor_angle_offset.detach().cpu().numpy().astype(float).item()
        self.params[field]['motors_vec'] = model.motors_vec.detach().cpu().numpy().tolist()
        self.params[field]['motors_pos'] = model.motors_pos.detach().cpu().numpy().tolist()
        self.params[field]['tau'] = model.tau.detach().cpu().numpy().astype(float).item()

    def save(self):        
        # Serializing json
        options = jsbeautifier.default_options()
        options.indent_size = 4
        json_object = jsbeautifier.beautify(json.dumps(self.params), options)
        
        with open(os.path.join(self.dir_path, 'params.json'), 'w') as outfile:
            outfile.write(json_object)
