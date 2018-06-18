# People Counting

By Prince Narula

Email: prince.narula222@gmail.com 

This repository provides an adaptation of the implementation of crowd counting described in the paper "Image Crowd Counting Using 
Convolutional Neural Network and Markov Random Field". I will demonstrate how to train and test this model on a new dataset. If 
you're here just to figure out how to annotate the data, jump straight ahead to the Data Annotation section.

 @article{han2017image,
  title={Image Crowd Counting Using Convolutional Neural Network and Markov Random Field},
  author={Han, Kang and Wan, Wanggen and Yao, Haiyan and Hou, Li},
  journal={arXiv preprint arXiv:1706.03686},
  year={2017}
 }


# Credits
Credits for the base code implementation of the paper go to https://github.com/hankong/crowd-counting. I have adapted the model by 
modifying files of the base code and adding new files for support.


# Dependencies
Appropriate C++ compiler and Python package should be installed.

The fully connected regress network is implemented using Keras with Tensorflow as backend. Other networks are implemented using MATLAB.

Make sure you install and compile MatConvNet before proceeding further.


# Getting Started 
1. Compile the MRF code by running 'testMRF.m' in the 'MRF'(MRF/) folder. If the code doesn't compile, you probably have an installation issue that needs attention.

2. If you wish, you may download the ShanghaiTech part B dataset using any of the following links:

   Dropbox: https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0
   
   Baidu Disk: http://pan.baidu.com/s/1nuAYslz

3. Place the ShanghaiTech part B dataset into the 'ShanghaiTech' folder of the project if you wish to use it. I have provided a mini 
dataset of Indian origin within 'ShanghaiTech' folder for reference so that you don't accidentally alter the configured paths which are 
used in the code implementation.

4. Download 'imagenet-resnet-152-dag.mat' and place it in the root folder of the project.

   Refer : http://www.vlfeat.org/matconvnet/pretrained/


# Data Annotation
I will demonstrate here how to annotate your own dataset and generate corresponding ground truth .mat files to make your data compatible 
with standard crowd counting models for training and testing. The format of the files generated here is same as that of the annotated 
ShanghaiTech dataset files. We desire that the dataset remains sequential. You may modify the steps according to your needs once you 
understand the procedure described below.

Let's say we wish to add our training dataset to the ShanghaiTech part B training dataset. The steps are as follows:

1. Go to 'data_annotation'(data_annotation/) folder.

2. Place your dataset images in the 'images'(data_annotation/images/) folder.

   a. Rename your images to 'IMG_num(t+i)' (for example, 'IMG_401') where:
   
              t - number of images already in the 'ShanghaiTech/part_B/train_data/images/' folder
              
              i - sequence number of an image in your training dataset (i=5 for 5th image)

3. Open 'gt_mat_gen.m'(data_annotation/gt_mat_gen.m), modify the value of t according to the purpose of t defined in step 2 and run this .m file using MATLAB.

4. Images will open sequentially as figures in MATLAB along with a marker(cursor). Use your mouse and click on the heads of the people to annotate the heads. Once you're done annotating all the heads in an image, press enter and wait for the next image figure to load. Repeat this step until you're done annotating all of your images.

5. Ground truth .mat files are generated in 'ground-truth'(data_annotation/ground-truth/) folder. Transfer the contents of this folder to 'ShanghaiTech/part_B/train_data/ground-truth/' folder. Also, images are resized to 1024*768 as this is the resolution of ShanghaiTech
part B images. You may change the desired resolution by changing appropriate parameters in the 'gt_mat_gen.m' file depending upon the model you're intending to use. Transfer the contents of 'images'(data_annotation/images/) folder to 'ShanghaiTech/part_B/train_data/images/' folder.

Note - If you're trying to add your testing dataset to the ShanghaiTech part B testing dataset, follow the same steps keeping in mind 
'ShanghaiTech/part_B/test_data/images/' instead of 'ShanghaiTech/part_B/train_data/images/' and 'ShanghaiTech/part_B/test_data/ground-truth/' instead of 'ShanghaiTech/part_B/train_data/ground-truth/'. Use t=0 if you're not using the ShanghaiTech dataset.


# Feature Extraction
1. Open MATLAB and compile MatConvNet.

   Refer : http://www.vlfeat.org/matconvnet/install/

2. Run 'DataPrep.m' to extract the following four files in 'data'(data/) folder required for training and testing the network.

   ground_truth_B_SHT.mat
   
   ground_truth_train_B_SHT.mat
   
   test_B_SHT.mat
   
   train_B_SHT.mat


# Training the model
Run 'regress_SHT.py'. After training, model architecture and weights are saved in the 'model' folder. 'predictions_B_SHT.mat' file for test data is extracted in 'data'(data/) folder which is needed for testing and evaluating the model.


# Using the saved trained model for extracting prediction file
Run 'patch_predict_SHT.py' to extract 'prediction_B_SHT.mat' file in 'data'(data/) folder which uses the test data features extracted by us during feature extraction.


# Testing and evaluating the model
Run 'EvaluateSHT.m'. Mean absolute error(MAE) and mean squared error(MSE) are displayed on console. True and predicted counts along with
test images are stored in 'result'(result/) folder in .png format.


# Testing and evaluating the saved trained model using provided extracted files
Directly run 'EvaluateSHT.m' without performing any of the above mentioned steps to evaluate the performance of the saved trained model 
using provided feature files in 'data'(data/) folder.


# Result
'result'(result/) folder contains the test images along with true count and predicted count. I have provided the result of training and testing the model using the mini dataset included in 'ShanghaiTech' folder.

# Getting the people count in a raw image
Go to 'raw'(raw/) folder. Place the raw images in 'image'(raw/images/) folder. Compile MatConvNet and run 'Prep.m'. Run 'predict.py',
followed by 'Evaluate.m'. Results are stored in 'output'(raw/output/) folder. I have provided some results for reference.

# Note
The extracted files provided in 'data'(data/) and 'model'(model/) folders correspond to the mini dataset of Indian origin in 'ShanghaiTech' folder. Hyperparameters used for training the model on this dataset are as follows:

Optimizer: Adam

Learning rate: 0.1

Epochs: 250

Batch size: 1000

Model performance on this mini dataset is as follows:

MAE: 3.125000

MSE: 3.889087

Tune the hyperparameters in 'regress_SHT.py' file if the model fails to fit well on your dataset.
