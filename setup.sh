git clone https://github.com/Replicable-MARL/MARLlib.git 
pip install -r ./MARLlib/requirements.txt
pip install "protobuf==3.20.0"
pip install "gym==0.20.0"  
cd ./MARLlib/marllib/patch && python add_patch.py -y
pip install --upgrade pip
pip install marllib
cp ./utils/machines.yaml ./venv/lib64/python3.8/site-packages/marllib/envs/base_env/config/machines.yaml
cp ./utils/postprocessing.py ./venv/lib/python3.8/site-packages/ray/rllib/evaluation/postprocessing.py