# For dev work:
./get_mt.sh

# preprocess the data into the ./data/build directory
cd data && sh run.sh

# Run
cd ..
python machine_translation.py --config config.json --gpu -1
