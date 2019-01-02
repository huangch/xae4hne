# eXclusive Autoencoder (XAE) for Nucleus Detection and Classification on Hematoxylin and Eosin (H&E) Stained Histopathological Images

This respository is the implementation of our paper published on (https://arxiv.org/abs/1811.11243). 

In this paper, we introduced a novel feature extraction approach, named exclusive autoencoder (XAE), which is a supervised version of autoencoder (AE), able to largely improve the performance of nucleus detection and classification on hematoxylin and eosin (H&E) histopathological images. The proposed XAE can be used in any AE-based algorithm, as long as the data labels are also provided in the feature extraction phase. In the experiments, we evaluated the performance of an approach which is the combination of an XAE and a fully connected neural network (FCN) and compared with some AE-based methods. For a nucleus detection problem (considered as a nucleus/non-nucleus classification problem) on breast cancer H&E images, the F-score of the proposed XAE+FCN approach achieved 96.64% while the state-of-the-art was at 84.49%. For nucleus classification on colorectal cancer H&E images, with the annotations of four categories of epithelial, inflammatory, fibroblast and miscellaneous nuclei. The F-score of the proposed method reached 70.4%. We also proposed a lymphocyte segmentation method. In the step of lymphocyte detection, we have compared with cutting-edge technology and gained improved performance from 90% to 98.67%. We also proposed an algorithm for lymphocyte segmentation based on nucleus detection and classification. The obtained Dice coefficient achieved 88.31% while the cutting-edge approach was at 74%.  

Please consider to include the following bibtex if this repository helps your publications:

@misc{huang2018exclusive,
    title={eXclusive Autoencoder (XAE) for Nucleus Detection and Classification on Hematoxylin and Eosin (H&E) Stained Histopathological Images},
    author={Chao-Hui Huang and Daniel Racoceanu},
    year={2018},
    eprint={1811.11243},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
