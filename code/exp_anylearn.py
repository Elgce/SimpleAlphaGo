import logging
logging.basicConfig(level=logging.INFO)
import argparse

from anylearn.applications.quickstart import quick_train
from anylearn.config import init_sdk

if __name__ == "__main__":
    args = parser.parse_args()
    init_sdk('https://anylearn.nelbds.cn', 'username', 'password')
    train_task, _, _, _ = quick_train(
        algorithm_dir="./",
        algorithm_name="AlphaZero",
        entrypoint="python main.py", # 训练启动命令，可以是python、shell等等
        quota_group_name='xxxxx',
        quota_group_request={
                'CPU': 8,
                'Memory': 32,
                "RTX-3090-shared": 1
            },
        hyperparams={},
        output="./temp",
        mirror_name="QUICKSTART_PYTORCH1.12.1_CUDA11",
        algorithm_force_update=True
    )
    print(train_task)
