**dependencies:**

- cuda 11.1
- torch 1.8.1 cuda version
- tensorflow 2.6.0 (for FVD calculation)
- tensorflow-gan 2.1.0 (for FVD calculation)
- scipy 1.6.3
- numpy 1.19.5
- imageio 2.9.0
- matplotlib 3.4.1


**This is the code for InterFormer on the DuetDance dataset.**

**WARNING : the code was tested on Windows, it should work on Macos and Linux but in case of trouble running it we recommend to use Windows**


**Contact the authors of the paper below to get acced to the skeleton data of the DuetDance dataset**
```
J. N. Kundu, H. Buckchash, P. Mandikal, R. M. V, A. Jamkhandi and R. V. Babu, "Cross-Conditioned Recurrent Networks for Long-Term Synthesis of Inter-Person Human Motion Interactions," 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass, CO, USA, 2020, pp. 2713-2722, doi: 10.1109/WACV45572.2020.9093627.
```

Then to format the data:
```
	copy the content of the database (male and female folder) into the 'data_raw' folder
	python format_files.py
```


To test our model on the DuetDance dataset use in a terminal:
```
	python test.py -load_weights Model -batch_size 16
	to generate the reactions without visuals
	
	python test.py -load_weights Model -batch_size 16 -visual
	to generate the reactions with visuals. Takes longer.
```

**IMPORTANT: the batch size of 16 works for a GPU with 8Go of video memory. Adapt the batch size according to your own GPU.**
	
After generating the reactions :
```
	python classifier.py
	to get the classification accuracy and get the deep features.
	
	python compute_FVD.py
	to get the FVD and diversity score. classifier.py must be used before to get the deep features
```	
	
Visuals are stored in the folders :
- "visual" for comparison with the ground truth (left GT, right generated. In blue the action, in the other color the reaction)
- "visual_single" for only the motion generated with InterFormer (in blue the action, in red the reaction)
	


Parts of the Interformer code are based on the code from  : https://github.com/SamLynnEvans/Transformer
The DeepGRU classifier code is based on : https://github.com/Maghoumi/DeepGRU
