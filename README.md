# InterFormer

Code for the paper "Interaction Transformer for Human ReactionGeneration" ([https://arxiv.org/pdf/2207.01685.pdf](https://arxiv.org/pdf/2207.01685.pdf))

InterFormer method to generate a reaction motion based on an action motion with 3D skeletons

![](https://github.com/CRISTAL-3DSAM/InterFormer/blob/main/Videos/SBU_Punching.mp4)

<video width="320" height="240" controls>
  <source src="https://github.com/CRISTAL-3DSAM/InterFormer/blob/main/Videos/SBU_Punching.mp4" type="video/mp4">
</video>


dependencies:
cuda 11.1
torch 1.8.1 cuda version
tensorflow 2.6.0 (for FVD calculation)
tensorflow-gan 2.1.0 (for FVD calculation)
scipy 1.6.3
numpy 1.19.5
imageio 2.9.0
matplotlib 3.4.1

Folder Interformer-SBU contains the code to run on the SBU interaction dataset ([link](https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html))

Folder Interformer-K3HI contains the code to run on the K3HI dataset([link](http://www.lmars.whu.edu.cn/prof_web/zhuxinyan/DataSetPublish/dataset.html))

Code for DuetDance will be available soon.

![alt text](https://github.com/CRISTAL-3DSAM/InterFormer/blob/main/Figures/Interformer(1).jpg "InterFormer overview")

