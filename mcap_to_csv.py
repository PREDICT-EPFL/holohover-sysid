# This file is part of holohover-sysid.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

# pip3 install pandas mcap-ros2-support
import os
import sys
import pandas as pd

from src.utils import read_mcap_file

def main():

    # maps topic to csv file
    topic_mapping = None
    # topic_mapping = {
    #     '/car/state': 'state',
    #     '/car/set/control': 'control',
    # }

    topic_msgs = read_mcap_file(sys.argv[1], topic_mapping.keys() if topic_mapping is not None else None)

    if topic_mapping is None:
        class TopicToFile:
            def __getitem__(self, topic):
                return topic.strip('/').replace('/', '_')
        topic_mapping = TopicToFile()

    dir = os.path.dirname(sys.argv[1])

    for topic, msgs in topic_msgs.items():
        df = pd.DataFrame(msgs)
        df.to_csv(os.path.join(dir, f'{topic_mapping[topic]}.csv'), index=False)

if __name__ == '__main__':
    main()
