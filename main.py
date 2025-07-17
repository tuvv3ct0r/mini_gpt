import argparse
import os
import yaml
import torch
from preprocessing import TextPreprocessor, CharDataModule
from model.gpt import GPT
from trainer import Trainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Set device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        config['device'] = 'cpu'
    return config

def main():
    parser = argparse.ArgumentParser(description='mini-gpt')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Path to config YAML')
    parser.add_argument('--sample', type=str, default=None, help='Sample: "start_text,length"')
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = 'data'
    input_path = config['input_txt']

    # Preprocessing
    preproc = TextPreprocessor(input_path, data_dir)
    preproc.download()
    if not (os.path.exists(os.path.join(data_dir, 'train.bin')) and os.path.exists(os.path.join(data_dir, 'val.bin'))):
        meta = preproc.preprocess()
    else:
        meta = preproc.load_meta()
    config['vocab_size'] = meta['vocab_size']

    # Model, DataModule
    model = GPT(type('Config', (), config))
    datamodule = CharDataModule(data_dir, config['batch_size'], config['max_seq_len'])
    trainer = Trainer(model, datamodule, config, log_dir=config['log_dir'], ckpt_dir=config['ckpt_dir'])

    if args.train:
        trainer.train()
    elif args.sample:
        start_text, length = args.sample.split(',')
        length = int(length)
        print(trainer.sample(start_text, length, meta['stoi'], meta['itos']))
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 