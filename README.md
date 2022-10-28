# Problem understanding: Cat vs. Classifier #
In this toy project, we aim to build a deep learning clssifier that predicts whether an image is cat or not. </br>

<!-- <img src="/DL_Projects/images/cat.0.jpg" align="center">
<img src="/DL_Projects/images/dog.0.jpg" align="right"> -->


  Cat            |  Dog
:--------------:|:--------:
![](figures/cat.4.jpg)  |  ![](figures/dog.4.jpg)



<!-- <br> -->
Refer to this [link](https://www.kaggle.com/competitions/dogs-vs-cats/data) to get the data.

# Run the project #
In this project, we have two steps: training and predicting. In the predict step, you can upload any image from your laptop and predict it. Let's show you how to run the project.

If you do not have venv package, please refer to this [link](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/)
</br>

## Create virtual environment ##

```
$ python3 -m venv ENV_NAME
```
## Activate your environment ##

```
$ source ENV_NAME/bin/activate
```

## Requirement installations ##
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
# Training the model #

```
$ python3 main.py --model MODEL_NAME --num_epochs
```
## Example of running models ##

```
$ python3 main.py --model resnet --10
```

```
$ python3 main.py --model cnn --10
```

