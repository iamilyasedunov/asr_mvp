# Prepare
```
python3.8 -m venv venv_exp
source venv_exp/bin/activate

python3.8 -m pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate torchaudio

```

# Modify manual
```
usage: main.py modify [-h] [--speed SPEED] [--volume VOLUME] input_file output_file

positional arguments:
  input_file       Path to input WAV file
  output_file      Path to output modified WAV file

optional arguments:
  -h, --help       show this help message and exit
  --speed SPEED    Speed factor (default: 1.0)
  --volume VOLUME  Volume change in dB (default: 0)
```

# Transcribe manual
```
usage: main.py transcribe [-h] [--lang {russian,english}] input_file

positional arguments:
  input_file            Path to input WAV file

optional arguments:
  -h, --help            show this help message and exit
  --lang {russian,english}
                        Language of the model (default: english)
```
