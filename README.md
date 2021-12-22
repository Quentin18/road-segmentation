# Machine Learning: Project 2 - Road Segmentation

Project 2 of the Machine Learning course given at the EPFL Fall 2021.

The goal of this project is to segment satellite images by detecting roads.
Our classifier consists of a convolutional neural network called UNet.

## Team members

- Quentin Deschamps
- Emilien Seiler
- Louis Le Guillouzic

## Results

Model | Data augmentation | Postprocessing | F1 score | Accuracy | Submission
--- | --- | --- | --- | --- | ---
UNet | Yes | Yes | 0.901 | 0.946 | [#169349](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/169349)
UNet | Yes | No | 0.900 | 0.945 | [#168760](https://www.aicrowd.com/0d812527-da37-4b16-b4e2-d01668a7573c)
Nested UNet | Yes | No | 0.896 | 0.943 | [#169077](https://www.aicrowd.com/413f9963-f51c-4b39-ad49-b7f4861a994c)
SegNet | Yes | No | 0.895 | 0.944 | [#169078](https://www.aicrowd.com/2f4c1472-85e2-467b-9091-19f96b796c5b)
UNet | No | No | 0.853 | 0.922 | [#169073](https://www.aicrowd.com/40e89fef-8084-40aa-9c29-46340fdc9f89)


## Installing

To run the code of this project, you need to install the libraries listed in
the `requirements.txt` file. You can perform the installation using this
command:
```
pip3 install -r requirements.txt
```

Dependencies:
- matplotlib
- numpy
- pillow
- scikit-image
- torch
- torchvision
- tqdm

## Instructions

The `scripts` directory contains scripts to perform the different tasks of the
project.

### Predictions for AIcrowd

To reproduce our submission on
[AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation), move
in the `scripts` folder and run:
```
python3 run.py
```
This command will create the predicted mask for each test image in the
`out/submission` directory. The csv file for submission produced will be
`out/submission.csv`.

### Data augmentation

To create the augmented training dataset, you can run:
```
python3 augment_data.py
```
The images created will be in the `data/training_augmented` directory. If this
directory already exists, it will overwrite the images.

### Training

To train a model, you can use the `train.py` script:
```
python3 train.py
```
To see the different options, run `python3 train.py --help`.

### Predicting

To create the predicted masks using a trained model, you can use the
`predict.py` script:
```
python3 predict.py
```
To see the different options, run `python3 predict.py --help`.

### Plotting

The `pickle` files created during training can be visualized using the
`plot_metrics.py` script:
```
python3 plot_metrics.py --file FILE
```
`FILE` must be a `pickle` file.

## Structure

This is the structure of the repository:

- `data`: contains the datasets
- `docs`: contains the documentation
- `figs`: contains the figures
- `notebooks`: contains the notebooks
- `scripts`: contains the main scripts
    - `augment_data.py`: create the augmented dataset
    - `config.py`: helpers functions to configure paths
    - `plot_metrics.py`: plot metrics
    - `predict.py`: make predictions using a trained model
    - `run.py`: make predictions for AIcrowd
    - `train.py`: train the model
- `src`: source code
    - `models`: neural network models
        - `nested_unet.py`: nested UNet implementation
        - `segnet.py`: SegNet implementation
        - `unet.py`: UNet implementation
    - `data_augmentation.py`: creation of the augmented dataset
    - `datasets.py`: custom dataset class for satellite images
    - `loss.py`: custom loss functions
    - `metrics.py`: score and performance functions
    - `path.py`: paths and archives management
    - `plot_utils.py`: plot utils using matplotlib
    - `postprocessing.py`: postprocessing functions to improve predictions
    - `predicter.py`: predicter class to make predictions using a trained model
    - `submission.py`: submission utils
    - `trainer.py`: trainer class to train a model

## References

See [references](references.md).
