# HiCervix (This paper is under review)
####

This repo contains the original source code of the paper _HiCervix: An Extensive Hierarchical Dataset and Benchmark for Cervical Cytology Classification_.

In this paper, we release the largest three-level hierarchical cervical dataset (HiCervix), and propose a hierarchical vision transformer-based classification benchmark method (HierSwin).

### HiCervix Dataset:

HiCervix includes 40,229 cervical cells and is categorized into 29 annotated classes.   These classes are organized within a three-level hierarchical tree
to capture fine-grained subtype information. 
![这是图片](figure1.png)


### HierSwin(benchmark method):
HierSwin is a hierarchical vision transformer-based classification network, where Swin transformer is first adopted for fine-grained feature extraction and a hierarchical classification head is integrated into the backbone network to merge the
information of fine-grained features.
<!-- The benchmark method of HierSwin and all the other methods implemented in this manuscript are organized in this repository. --> 
#### Installation

The implementation of HierSwin is based on the repo [making-better-mistakes](https://github.com/fiveai/making-better-mistakes), where we rewrote the dataset loader for HiCervix and [albumentations](https://albumentations.ai/) is utilized for image augmentations.


#### Dataset preparation
HiCervix can be downloaded from Zenodo (https://zenodo.org/records/xxx). The dataset is splitted into three parts: train, validation, and test, each associated with one CSV file.
```
.
├── test
├── test.csv
├── train
├── train.csv
├── val
└── val.csv
```

The ```{train,val,test}.csv``` labels the hierarchical names for HiCervix dataset as follow:
|image_name                              |class_name |class_id|level_1|level_2|level_3    |
|----------------------------------------|-----------|--------|-------|-------|-----------|
|xxx.jpg|AGC-EMC-NOS|18      |AGC    |AGC-NOS|AGC-EMC-NOS|

The full names for the acronyms in the CSV files can be referenced in the ```dataset/hierarchy_classification/version2023
/hierarchy_names.csv```
<!--If you want to request data, please send me [data use agreement](https://docs.google.com/document/d/1B0fRRf8H40zG7l4gMnEUmr9PJaz5Z8HR/edit?usp=sharing&ouid=104345779948250629209&rtpof=true&sd=true) to this email (ys810137152@gmail.com) and we will send you the data link in 1-3 business days.-->

The hierarchical tree structure and lowest-common-tree distances files were generated beforehand and placed in the ```HierSwin/data``` as follow:

```
.
├── level_names_dict.pkl
├── tct_distances.pkl
└── tct_tree.pkl
```
For more details on the preprocessing of the Hierarchical structure of HiCervix, please see the ``` HiCervix_pre-processing.ipynb```
#### Training and evaluation
* Training of HierSwin, 
```
python3 scripts/start_training.py --arch swinT 
        --loss hierarchical-cross-entropy 
        --alpha 0.4 
        --output hierswin_alpha0.4 
        --train-csv train_hierswin.csv
        --val-csv val_hierswin.csv 
        --epochs 50
```

* Evaluation of HierSwin

```
python3 scripts/start_testing.py 
        --experiments_json hierswin_alpha0.4/opts.json 
        --checkpoint_path hierswin_alpha0.4/model_snapshots/checkpoint.epoch0050.pth.tar 
        --test-csv test_hierswin.csv
```
* The HierSwin and other methods usually will ouput the probabilities for finest level's hierarchy, the postprocess of the original evaluation results for HierSwin and other methods can be referenced in ```HiCervix_evaluation.ipynb```.




## License

HiCervix and HierSwin are released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this paper if you find our work useful in your research.
