# RoPPT
This repo contains the codes for the paper [Metaphor Detection with Effective Context Denoising](https://arxiv.org/abs/2302.05611).
Some codes are borrowed from [https://github.com/jin530/MelBERT](https://github.com/jin530/MelBERT).

## Requirements
* Python 3.7.7

## Environment Set Up and Data Preprocessing
```
sh env_setup.sh
sh parser.sh
```

## Data
We use four well-known public English datasets. The VU Amsterdam Metaphor Corpus (VUA) has been released in metaphor detection shared tasks in 2018 and 2020. We use two versions of VUA datasets, called **VUA-18** and **VUA-20**, where VUA-20 is the extension of VUA-18. 
We employ **MOH-X** and **TroFi** for testing only.

We conducted experiments on dependency tree parsed data, which use two different parsers **Biaffine** and **spaCy**. However, the choice of parser has a relatively small impact on our model's performance. Therefore, we only keep and upload the preprocessed data parsed by **spaCy**.

You can get the raw and preprocessed data from this [link](https://drive.google.com/drive/folders/1JoVZlZQBbjBVVPmjtjfCveCZvIqVDgmq?usp=share_link).

## Run
You can run with this .sh script
```
sh scripts/run.sh
```
Or you can directly run this
```
python main.py --data_dir data_sample/VUA --task_name sample --model_type MELBERT_GAT --class_weight 3 --bert_model roberta-base --num_train_epoch 3 --train_batch_size 8 --learning_rate 3e-5 --warmup_epoch 2
```
