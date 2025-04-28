import sys# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)


from x_core_ai.src.core import Core
from sub_module.utilx.src.config import ConfigLoader
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/multi_task_vit_v1.0.0.yaml")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def debug(args):
    args.config = "configs/models/multi_task_vit_v1.0.0.yaml"

def main(args):
    config = ConfigLoader.load_config(args.config)
    core = Core(config)
    core.model_generator()
    core.model_to_device()
    core.model.eval()

    df_train, df_val, df_test = core.create_dataframes()

    train_dataset = core.create_dataset(df_train, train=True)
    val_dataset = core.create_dataset(df_val, train=False)

    print("iterating over train dataset")
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        print(f"{i}/{len(train_dataset)}")

    print("iterating over val dataset")
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        print(f"{i}/{len(val_dataset)}")    
    
    dataloader = core.create_dataloader(train_dataset, train=True)
    print("iterating over train dataloader")
    for batch in dataloader:
        print(batch)
        break

    dataloader = core.create_dataloader(val_dataset, train=False)
    print("iterating over val dataloader")
    for batch in dataloader:
        print(batch)
        break



if __name__ == "__main__":
    args = parse_args()
    args.debug = True
    if args.debug:
        debug(args)
    main(args)