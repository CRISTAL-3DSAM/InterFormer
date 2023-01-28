To test our model on the K3HI dataset use in a terminal:

	python test.py -load_weights Model -batch_size 2
	to generate the reactions without visuals
	
	python test.py -load_weights Model -batch_size 2 -visual
	to generate the reactions with visuals. Takes longer.
	
	IMPORTANT: the batch size of 2 works for a GPU with 8Go of video memory. Adapt the batch size according to your own GPU.
	
After generating the reactions :

	python classifier.py
	to get the classification accuracy and get the deep features.
	
	python compute_FVD.py
	to get the FVD and diversity score. classifier.py must be used before to get the deep features
	
	
Visuals are stored in the folders :
	- visual for comparison with the ground truth (left GT, right generated. In blue the action, in the other color the reaction)
	- visual single for only the motion generated with InterFormer (in blue the action, in red the reaction)
	


Parts of the Interformer code are based on the code from  : https://github.com/SamLynnEvans/Transformer
The DeepGRU classifier code is based on : https://github.com/Maghoumi/DeepGRU
