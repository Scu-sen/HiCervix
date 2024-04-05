
# HiCervix (This paper is under review)
####

This repo contains the original source code of the paper _HiCervix: An Extensive Hierarchical Dataset and Benchmark for Cervical Cytology Classification_.

In this paper, we release the largest three-level hierarchical cervical dataset (HiCervix), and propose a hierarchical vision transformer-based classification benchmark method (HierSwin).

### HiCervix Dataset:

HiCervix includes 40,229 cervical cells and is categorized into 29 annotated classes.   These classes are organized within a three-level hierarchical tree
to capture fine-grained subtype information. 
![这是图片](figure1.png)
HiCervix can be downloaded from Zenodo (https://zenodo.org/records/xxx). The dataset is splitted into three parts: train, validation, and test, each associated with one csv file to label the hierarchical names as follow:
|image_name                              |class_name |class_id|level_1|level_2|level_3    |
|----------------------------------------|-----------|--------|-------|-------|-----------|
|xxx.jpg|AGC-EMC-NOS|18      |AGC    |AGC-NOS|AGC-EMC-NOS|
<!--If you want to request data, please send me [data use agreement](https://docs.google.com/document/d/1B0fRRf8H40zG7l4gMnEUmr9PJaz5Z8HR/edit?usp=sharing&ouid=104345779948250629209&rtpof=true&sd=true) to this email (ys810137152@gmail.com) and we will send you the data link in 1-3 business days.-->


### HierSwin(benchmark method):
HierSwin is a hierarchical vision transformer-based classification network, where Swin transformer is first adopted for fine-grained feature extraction and a hierarchical classification head is integrated into the backbone network to merge the
information of fine-grained features.
<!-- The benchmark method of HierSwin and all the other methods implemented in this manuscript are organized in this repository. --> 
The implementation of HierSwin is based on the repo https://github.com/fiveai/making-better-mistakes.
* For the training of HierSwin, 

```
python3 scripts/start_training.py
```

* For the evaluation of HierSwin

```
python3 scripts/start_testing.py
```
* The postprocess of the original evaluation results for HierSwin and other methods
```
HiCervix_processing.ipynb
```

## License

HiCervix and HierSwin are released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this paper if you find our work useful in your research.

