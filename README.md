<!-- # MoCHA-former:  -->

# Version Info

* basicsr==1.4.2
* scikit-image==0.15.0
* deepspeed

# Environment Setting

We provide two types of environments: a Conda environment and a Docker-based environment.

## Using Conda
```
conda env export -n RRID environments.yml
```

## Using Docker
1. downloading docker image
```
docker pull jeahun00/rrvd:latest
```

2. executing docker container 
```
docker container run -it -d --gpus all --shm-size=128G -v {local_server_path}:/code --name demoire jeahun00/rrvd:latest /bin/bash
```

3. enter docker container
```
docker exec -it demoire /bin/bash
```

4. activating anaconda environment
When using Docker, one can utilize RRID within the Conda environment inside the container.
```
conda activate RRID
```

# Prepare Dataset

* [RawVDemoire full dataset link](https://github.com/tju-chengyijia/VD_raw)
* [RawVDemoiré testset link](https://drive.google.com/file/d/1iviOHDjTWIxgo3DMs4kQJWn7dqeVW-CY/view?usp=sharing).
* unzip downloaded file

```
unzip ./Rawvdemoire_test.zip
```



<!-- /home/jeahun/demoire_ai/options/test/demoire_ai_01.yml -->
# Executing Command

You must adjust `dataroot_gt` and `dataroot_lq` in each corresponding YAML file for both training and testing to match the paths of your own dataset.

# Example of Training Command

```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/demoire_ai_01.yml
```

# Example of Inference Command

```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/demoire_ai_01.yml
```

