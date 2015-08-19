#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading raw data"

mkdir -p data && cd data
wget russellsstewart.com/s/mt/run.sh
wget russellsstewart.com/s/mt/un.en-es.tgz
tar xf un.en-es.tgz

echo "Done."
