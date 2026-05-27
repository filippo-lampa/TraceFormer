# :loudspeaker: TraceFormer

Transformer-based anomaly detection for smart contract execution traces.

## :zap: Quick Start

### :envelope_with_arrow: 1. Download the data

Download the dataset from [this link](https://drive.google.com/drive/folders/1qg6wA1EulpqzX_CYeJZv3ZmgWTkt5CCQ?usp=sharing) and place the contents in the `data/` directory.

### :package: 2. Install dependencies

```bash
pip install -r requirements.txt
```

### :rocket: 3. Run

From the project root (`TraceFormer/`):

```bash
# Train only
python main.py --train

# Test only
python main.py --test

# Train and test together
python main.py --train --test
```

### :open_book: 4. Requirements

- Python 3.8+

- See `requirements.txt` for full dependencies
