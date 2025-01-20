# Geo-Encoder: A Chunk-Argument Bi-Encoder Framework for Chinese Geographic Re-Ranking

This repository is the source code for the paper ["Geo-Encoder: A Chunk-Argument Bi-Encoder Framework for Chinese Geographic Re-Ranking, EACL 2024"](https://arxiv.org/pdf/2309.01606).


## Usage

### Training
To train the model, use the following command:
```bash
sbatch script/train.sh
```
The training script is located in `script/train.sh` and can be customized to fit your computational environment.

### Evaluation
After training, evaluate the model using:
```bash
sbatch script/eval.sh
```

### Configuration
Hyperparameters and other configurations are managed in `config.yaml`. Adjust these settings as needed for your experiments.

## Dataset
The datasets used in this project are specifically curated for geographic re-ranking tasks. These datasets include Chinese text with annotated geographic references.

To access the datasets, please refer to the `data/` directory. Ensure the data is preprocessed correctly before training.

## Citation
If you use this code or find this project helpful, please consider citing our paper:
```
@inproceedings{cao-etal-2024-geo,
    title = "Geo-Encoder: A Chunk-Argument Bi-Encoder Framework for {C}hinese Geographic Re-Ranking",
    author = "Cao, Yong  and
      Ding, Ruixue  and
      Chen, Boli  and
      Li, Xianzhi  and
      Chen, Min  and
      Hershcovich, Daniel  and
      Xie, Pengjun  and
      Huang, Fei",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.91/",
    pages = "1516--1530"
}
```

## Contact
If you have any questions or feedback, please feel free to reach out:
- Email: yongcao2018@gmail.com

