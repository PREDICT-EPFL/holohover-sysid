# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from collections import defaultdict
from mcap_ros2.reader import read_ros2_messages

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def read_mcap_file(path, topics=None):
    def get_attributes(obj):
        return filter(lambda a: not a.startswith('_'), dir(obj))

    def msg_to_dic(msg):
        dic = {'log_time_ns': msg.log_time_ns}
        
        def recursive_traversal(prefix, obj, depth):
            if depth > 5: return
            for attr in get_attributes(obj):

                field = getattr(obj, attr)
                
                if prefix == '':
                    name = attr
                else:
                    name = f'{prefix}.{attr}'

                if type(field).__name__ == 'Quaternion':
                    roll, pitch, yaw = euler_from_quaternion(field.x, field.y, field.z, field.w)
                    dic[f'{name}.roll'] = roll
                    dic[f'{name}.pitch'] = pitch
                    dic[f'{name}.yaw'] = yaw

                if type(field) in [bool, int, float, str]:
                    dic[name] = field
                elif type(field) not in [list]:
                    recursive_traversal(name, field, depth + 1)
                    
        recursive_traversal('', msg.ros_msg, 0)
        return dic

    topic_msgs = defaultdict(list)
    for msg in read_ros2_messages(path, topics):
        topic_msgs[msg.channel.topic].append(msg_to_dic(msg))

    return topic_msgs

def poly_expand_u(U, deg):
    """
    Make polynomial expansion of control input
    Args:
        U: control input, tensor (N,M)                        
    Returns:
        Upoly: polynomial expansion of U, if poly_expand_U=2 and U=[u1,u2,u3] 
                then Upoly=[u1*u1, u2*u2, u3*u3], tensor (N,poly_expand_U*M)
    """
    U = U.reshape(*U.shape, 1)

    Upoly = U
    for deg in range(2, deg + 1):
        Upoly = torch.cat((Upoly, torch.pow(U, deg)), dim=-1)

    return Upoly.flatten(start_dim=-2, end_dim=-1)

if __name__ == '__main__':
    
    U = torch.rand((3, 6))
    print(U)
    print(poly_expand_u(U, 3))
    print(poly_expand_u(U, 3).shape)
