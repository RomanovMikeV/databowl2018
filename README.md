# databowl2018

Repository for competition https://www.kaggle.com/c/data-science-bowl-2018

## Requirements

To run this code you need

- pytorch installed with CUDA support
- kaggle-cli installed and tuned

## Download and unpack data

In the data directory run

'''
$bash get_data.py
'''

## Run the bounding boxer training

Run this to start training the boxer part of the model:
'''
cd train
python train.py
'''

You can do that until you are satisfied with the results.
For the final tuning run

'''
python train_stage2.py
'''

## Run the masker training

To start training the masker part of the model run this:
'''
python train_masker.py
'''

Here are some of the results:

