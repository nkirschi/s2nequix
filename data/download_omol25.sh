#!/bin/bash

wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train.tar.gz -P data/
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz -P data/
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz -P data/
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/test.tar.gz -P data/


tar -xzf data/train.tar.gz
tar -xzf data/train_4M.tar.gz
tar -xzf data/val.tar.gz
tar -xzf data/test.tar.gz