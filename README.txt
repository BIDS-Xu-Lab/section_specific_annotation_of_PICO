This file briefly describes the steps to train/predict PICO entities.
1. Create virtual environment using conda.
$ conda create -n litcoin python=3.7
$ conda activate litcoin

2. Install PyTorch.
Using Cuda, please choose compatible cuda version with torch (reference: https://pytorch.org/get-started/previous-versions/):
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
Without using Cuda
$ conda install pytorch cpuonly -c pytorch

3. Install transformers.
$ pip install transformers==4.6.0

4. Install seqeval
$ pip install seqeval

5. Install tensorboardX
$ pip install tensorboardX

6. config PICO_ner.py for input data dir, model dir, and output data dir.

7. Run PICO_ner.py to train the model and predict on the test dataset.
