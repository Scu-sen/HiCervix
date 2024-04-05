
# HiCervix (This paper is under review)
####

We release the largest three-level hierarchical cervical dataset (HiCervix), which can be downloaded from Zenodo (https://zenodo.org/records/xxx). The dataset is splitted into three parts: train, validation, and test, each associated with one csv file to label the hierarchical names as follow:
|image_name                              |class_name |class_id|level_1|level_2|level_3    |
|----------------------------------------|-----------|--------|-------|-------|-----------|
|xxx.jpg|AGC-EMC-NOS|18      |AGC    |AGC-NOS|AGC-EMC-NOS|
<!--If you want to request data, please send me [data use agreement](https://docs.google.com/document/d/1B0fRRf8H40zG7l4gMnEUmr9PJaz5Z8HR/edit?usp=sharing&ouid=104345779948250629209&rtpof=true&sd=true) to this email (ys810137152@gmail.com) and we will send you the data link in 1-3 business days.-->


### HierSwin(benchmark method):
The benchmark method of HierSwin and all the other methods implemented in this manuscript are organized in this repository.  
* For the training of HierSwin, 

```
cd experiments
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


