# MGIPF
The repo for "MGIPF: Multi-Granularity Interest Prediction Framework for Personalized Recommendation", SIGIR 2025



## Install

```
conda create -n mgipf python=3.8
conda activate mgipf
pip install -r requirements.txt
```



## Data Preparation

- Download and unzip 'UserBehavior.csv.zip' (https://tianchi.aliyun.com/dataset/649) to `data_user/`
- Download and unzip 'data_format1.zip' (https://tianchi.aliyun.com/dataset/42) to `data_ijcai/`
- Download the 4 files of Taobao (https://tianchi.aliyun.com/dataset/56) and unzip to `data_taobao/raw_data/`



## Run (Data Process + Train)

1. Modify the 'dataset' in `run.sh`
2. Run `run.sh` and wait



## Citation

```
@inproceedings{feng2025mgipf,
  title={MGIPF: Multi-Granularity Interest Prediction Framework for Personalized Recommendation},
  author={Feng, Ruoxuan and Tian, Zhen and Peng, Qiushi and Mao, Jiaxin and Zhao, Wayne Xin and Hu, Di and Zhang, Changwang},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2172--2181},
  year={2025}
}
```

