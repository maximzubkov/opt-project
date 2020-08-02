#!/bin/bash

echo "default_gan, n_cr=2"
python train.py default_gan --n_cr=2
echo "default_gan, n_cr=4"
python train.py default_gan --n_cr=4
echo "default_gan, n_cr=6"
python train.py default_gan --n_cr=6
echo "sn_gan, n_cr=2"
python train.py sn_gan --n_cr=2
echo "sn_gan, n_cr=4"
python train.py sn_gan --n_cr=4
echo "sn_gan, n_cr=6"
python train.py sn_gan --n_cr=6
