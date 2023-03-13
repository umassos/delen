#!/usr/bin/env sh

wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir speech_commands

tar -xf speech_commands_v0.02.tar.gz -C speech_commands