import sys# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

from x_core_ai.src.core import Forecast    
from sub_module.utilx.src.config import ConfigLoader
from argparse import ArgumentParser
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/multi_task_vit_v1.0.0.yaml")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def debug(args):
    args.config = "configs/models/multi_task_vit_v1.0.0.yaml"

def main(args):
    config = ConfigLoader.load_config(args.config)
    forecast = Forecast(config)
    demo_input = {
        'images': np.random.randn(1, 4, 3, 224, 224)}
    
    outputs = forecast.predict(demo_input)
    # convert the token ids to string
    print("\n\n Estimated title:")
    print(outputs)
    print("\n\n")

    # batch predict
    demo_input = [
        {
            'images': np.random.randn(1, 4, 3, 224, 224)
        }
        for _ in range(4)
    ]
    
    batch_predictions = forecast.batch_predict(list_of_dicts=demo_input)
    print(batch_predictions)
    print("\n\n")

    # TODO: batch predict with dataloader


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)
    main(args)


