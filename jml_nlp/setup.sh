#!/bin/bash

#curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

if [ ! -d "./ner_tool" ]; then
    git clone https://gitlab.ow2.org/decoder/ner_tool.git
fi

if [ ! -d "./srl_tool" ]; then
    git clone https://gitlab.ow2.org/decoder/srl_tool.git
    cd srl_tool && git lfs pull && cd ..
fi

sudo docker-compose --env-file ./nlp.env build
sudo docker-compose --env-file ./nlp.env up