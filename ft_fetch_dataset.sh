#!/bin/bash

# download dataset
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
mv images dataset
mkdir dataset/Apple dataset/Grape
mv dataset/Apple_* dataset/Apple
mv dataset/Grape_* dataset/Grape

rm leaves.zip
