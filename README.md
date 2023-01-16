# Project: Dog Breed Classifier

## Summary:
This project is part of the Udacity Data Scientis Nanodegree. A full blog post detailing the project can be found here: https://medium.com/@Dennis.Burckhardt/how-to-become-a-dog-expert-in-5-minutes-e3f5567b666b

The goal of this project is to apply the data science skills learned in the course to build a CNN which can classify the breed of a dog from a given picture. Furthermore if a given picture shows a human, the model classifies the dog breed that is the most ressembling.

![Alt text](https://github.com/Deburckh/Udacity_Capstone_Project_Dog_Classifier/blob/main/images/Labrador_retriever_06449.jpg?raw=true "Title")


## Files 

The project mainly consists of a jupyter notebook:

dog_app.ipynb
The images and test images for the notebook and the testing of the algorithm can be found in the images and test_images folders.

Following files have to be imported as modules within the notebook:

extract_bottleneck_features.py
The Haar feature-based cascade classifiers for face detection can be found in the haarcascades folder

The folder saved_models contains the trained models, that I achieved while working on the project:

ResNet50_model.h5 and weights.best.ResNet50_0.3_1024.hdf5 (ResNet50 models with highest accuracy achieved)
weights.best.VGG16.hdf5(VGG16 model of step 4 of the notebook)
weights.best.VGG19_0.15_256.hdf5(example of VGG19 model with an added dense layer with 256 nodes and a dropout layer with a dopout rate of 0.15)


## Libraries & Installations:
A full list of requirements can be found under `requirements/requirements.txt`

To install all Python packages in the `requirements.txt` file run `pip install -r requirements.txt`

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

## Acknowledgements

- Data Source and Project Idea: Udacity Data Science Nanodegree
- Author: Dennis Burckhardt
- License: MIT License
