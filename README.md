# Graph-Structure

**Valentin Bencheci**

**Melissa Bouamama**

**Yann Le Lorier**

## AMAL Final project - Sorbonne Université

Unofficial implementation of the [GRAPH-BASED DOCUMENT STRUCTURE ANALYSIS](https://arxiv.org/pdf/2502.02501) paper, accepted paper for ICLR 2025.

This project is applied to the Visual Genome dataset.


## Generate the database

First, download data from the [Visual Genome organization](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html).

Specifically:
- [Objects](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip)
- [Relationships](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip)
- [Image Metadata](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)
- [Images (part 1)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) only

And place these files in a directory called `data/`, at the same level as `generateh5.py`.

Now generate the h5 file:

```sh
python generateh5.py
```

The output (`VG-SGG.h5`) will be the file used for training.

### Dictionary

For the dictionary mapping indices to objects (`VG-SGG-dicts`) download from [here](https://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json).

## Train!

Now you can run:

```sh
python main.py
```

## Inference

Currently, the inference is harcoded to image `15.jpg` in the `VG100K` directory. Feel free to change the image used.

```sh
python inference.py
```
