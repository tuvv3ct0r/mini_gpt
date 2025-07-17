# mini_gpt
A character-level implementation of GPT

## Setup

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Train the model
```
python main.py --train --config config/config.yml
```

### Sample from the model
```
python main.py --sample "Once upon a time,100" --config config/config.yml
```

- All settings are in `config/config.yml`.
- Logs are saved to the `logs/` directory (view with TensorBoard).
- Model checkpoints and loss plots are saved to `checkpoints/`.
