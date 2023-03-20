# InterFormer

Code for the paper "Interaction Transformer for Human ReactionGeneration" ([https://arxiv.org/pdf/2207.01685.pdf](https://arxiv.org/pdf/2207.01685.pdf))

![](https://github.com/CRISTAL-3DSAM/InterFormer/blob/main/Figures/Interformer(1).jpg "InterFormer overview")

InterFormer method to generate a reaction motion based on an action motion with 3D skeletons

https://user-images.githubusercontent.com/105372137/225047449-5b961338-ce44-4dbb-ae78-0cb7e99b7676.mp4


**dependencies:**
- cuda 11.1
- torch 1.8.1 cuda version
- tensorflow 2.6.0 (for FVD calculation)
- tensorflow-gan 2.1.0 (for FVD calculation)
- scipy 1.6.3
- numpy 1.19.5
- imageio 2.9.0
- matplotlib 3.4.1

Folder Interformer-SBU contains the code to run on the SBU interaction dataset ([link](https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html))

Folder Interformer-K3HI contains the code to run on the K3HI dataset([link](http://www.lmars.whu.edu.cn/prof_web/zhuxinyan/DataSetPublish/dataset.html))

Code for DuetDance will be available soon.

**Detailed instructions to run the code are available in each folder** 


**Citation:**
```
@ARTICLE{10036100,
  author={Chopin, Baptiste and Tang, Hao and Otberdout, Naima and Daoudi, Mohamed and Sebe, Nicu},
  journal={IEEE Transactions on Multimedia}, 
  title={Interaction Transformer for Human Reaction Generation}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMM.2023.3242152}}
```

**Acknowledgements**

This project has received financial support from the CNRS through the 80—Prime program, from the French State, managed by the National Agency for Research (ANR) under the Investments for the future program with reference ANR-16-IDEX-0004 ULNE and by the EU H2020 project AI4Media under Grant 951911. 
