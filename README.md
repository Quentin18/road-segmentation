# Machine Learning: Project 2 - Road Segmentation

Project 2 of the Machine Learning course given at the EPFL Fall 2021.

## Team members

- Quentin Deschamps
- Emilien Seiler
- Louis Le Guillouzic

## Installing

To run the code of this project, you need to install the libraries listed in
the `requirements.txt` file. You can perform the installation using this
command:
```
pip3 install -r requirements.txt
```

## Instructions

### Predictions for AIcrowd

To reproduce our submission on
[AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation), move
in the `scripts` folder and run:
```
python3 run.py
```
The csv file produced will be `out/predictions.csv`.

## Structure

This is the structure of the repository:

- `data`: contains the datasets
- `docs`: contains the documentation
- `figs`: contains the figures
- `notebooks`: contains the notebooks
- `scripts`: contains the main scripts
    - `config.py`: helpers functions to configure paths
    - `predict.py`: make predictions using a trained model
    - `run.py`: make predictions for AIcrowd
    - `train.py`: train the model
- `src`: source code
    - `models`: neural network models
        - `segnet.py`: SegNet implementation
        - `unet.py`: UNet implementation
    - `datasets.py`: custom dataset class for satellite images
    - `metrics.py`: score and performance functions
    - `path.py`: paths and archives management
    - `plot_utils.py`: plot utils using matplotlib
    - `predicter.py`: predicter class to make predictions using a trained model
    - `submission.py`: submission utils
    - `trainer.py`: trainer class to train a model

## References

See [references](references.md).
