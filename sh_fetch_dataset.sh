#!/bin/bash

# download dataset
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
mv images dataset
rm leaves.zip
