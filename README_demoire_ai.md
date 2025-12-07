# 환경설정

* conda, docker 등을 모두 지원할 수 있도록 `README.md` 에 제공하였다.
* 하지만 docker 환경이 설치가 수월하여 해당환경을 추천한다.


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
* 위의 환경에 따라 docker container 를 실행했을 때 local_server_path 가 mount folder 의 위치이다.
* 즉, 코드 zip 파일과 RawVDemoire testset 을 해당 폴더 하위에 두어야 접근이 가능하다.

# Prepare Dataset

* 데이터셋의 경우 full dataset 을 받아도 되지만 해당 데이터셋의 용량이 큰 편이라서 내가 지정한 drive link 의 testset 만 받아서 inference 만 수행하는 것을 추천한다.

* [RawVDemoire full dataset link](https://github.com/tju-chengyijia/VD_raw)
* [RawVDemoiré testset link](https://drive.google.com/file/d/1iviOHDjTWIxgo3DMs4kQJWn7dqeVW-CY/view?usp=sharing).
* unzip downloaded file

```
unzip ./Rawvdemoire_test.zip
```

<!-- /home/jeahun/demoire_ai/options/test/demoire_ai_01.yml -->
# Executing Command

You must adjust `dataroot_gt` and `dataroot_lq` in each corresponding YAML file for both training and testing to match the paths of your own dataset.

# Example of Inference Command
* 위에서도 언급하였듯, `options/test/demoire_ai_01.yml` 파일을 수정해야 한다.

```yaml
name: demoire_ai_01
model_type: MoCHAformer_Stg1_Model
num_gpu: 1  # set num_gpu: 0 for cpu mode
manual_seed: 0

datasets:
  test:
    name: MultiFrameVD
    type: MultiFrameVDPairedRawTestDataset
    dataroot_gt: /code/RawVDemoire_test/gt_rgb # <--- 여기를 내가 준 Rawvdemoire testset 경로에 맞게 수정해야 함
    dataroot_lq: /code/RawVDemoire_test/moire_raw # <--- 여기를 내가 준 Rawvdemoire testset 경로에 맞게 수정해야 함
    io_backend:
      type: disk
    batch_size: 1
    num_worker: 8

# network structures
network_g:
  type: MoCHAformer4
  stage: 1
  convnext_block_c: 3
  convnext_block_m: 3
  img_size: 128
  patch_size: 1
  in_chans: 4
  embed_dim: 96
  depths: [3,3,3]
  num_heads: [4,4,4]
  window_size: [2,7,7]
  mlp_ratio: 2
  qkv_bias: True
  qk_scale: False
  drop_rate: 0.
  attn_drop_rate: 0.
  drop_path_rate: 0.1
  ape: False
  patch_norm: True
  use_checkpoint: False
  upscale: 1
  img_range: 1.
  upsampler: 'pixelshuffle'
  downsample: True


# path
path:
  pretrain_network_g: experiments/demoire_ai_01/models/net_g_120000.pth
  strict_load_g: false
  param_key_g: 'params_ema'


# validation settings
val:
  save_img: true
  metrics:
    psnr:
      type: calculate_vd_psnr
    ssim:
      type: calculate_vd_ssim
    lpips:
      type: calculate_vd_lpips
```

* 위의 경로를 수정하고 아래 명령어를 실행하면 문제 없이 동작할 것이다.

```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/demoire_ai_01.yml
```

* 위의 명령어를 실행하면 demoiréd frame 은 `results/demoire_ai_01/visualization/MultiFrameVD` 에 저장된다.
* 또한 최종 정량평가의 결과는 `results/demoire_ai_01/test_demoire_ai_01_20251207_165001.log` 에 저장된다.