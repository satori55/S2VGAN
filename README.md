# S2VGAN with U-Net Generator and Discriminator
（The complete code will be published after the complement of the paper publication）

Our research addresses critical challenges in microstructural characterization and optimization, which are essential across various chemical engineering applications, including catalysis, materials processing, and energy storage.

## Project Overview

This project implements a 3D GAN (Generative Adversarial Network) with a U-Net-based generator and a convolution discriminator. The model is designed for generating 3D volumetric data, such as medical images or other 3D datasets from several scanned material slices.

## File Descriptions

- **`generator.py`**: Contains the implementation of the 3D U-Net generator.
- **`discriminator.py`**: Contains the implementation of a multi-layer convolution discriminator used to evaluate the quality of generated 3D data.
- **`train.py`**: Contains the training logic, including optimization for both the generator and discriminator, and techniques like gradient penalty.
- **`rungan.py`**: Main entry point of the project, defines training and testing configurations, and invokes the training pipeline and utility functions.
- **`preprocessing.py`**: Preprocessing module supporting data loading, random batch generation, and multi-scale data augmentation.
- **`util.py`**: Provides utility functions for weight initialization, gradient penalty calculation, project path management, image postprocessing, and result visualization.


## Requirements

The following Python libraries are required:

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- tifffile
- torchvision

Install the required libraries using the following command:

```bash
pip install torch numpy matplotlib tifffile torchvision
```

## Usage Instructions

### 1. Data Preparation

Place the training data (.tif) in the specified path and update the `data_path` variable in `rungan.py` to point to your dataset.

### 2. Configure the Model

In `rungan.py`, adjust the following parameters:

- **Data Type**: Set `data_type` to match the format of your dataset.
- **Network Architecture**: Modify `feature_list_Enc` and `feature_list_Dec` to change the number of layers and channels in the generator.
- **Image Size**: Adjust `img_size` and `scale_factor` to fit the resolution of your data.

### 3. Train the Model

Run the following command to start training:

```bash
python rungan.py
```

The generator and discriminator weights will be saved to the specified path upon completion.

### 4. Test the Model

Set `Training` to `False` in `rungan.py`, and run the following command to generate samples:

```bash
python rungan.py
```


# Deep Learning FFc Prediction Model

## Project Overview

This project implements a deep learning model to predict Flow Function Coefficient (FFc) values using powder characteristics data. The model utilizes a hybrid architecture combining convolutional neural networks (CNN) and fully connected layers to process and analyze powder flow properties.

## Features

- Deep neural network combining CNN and fully connected layers
- Automated data preprocessing and normalization
- Real-time training monitoring with loss and R² metrics
- Model performance visualization
- Best model state saving based on R² score
- Prediction capability for new samples


## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

Install the required packages using:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```
## Data Format

The input data should be in Excel (.xlsx) format with the following columns:
- D10 (μm)
- D50 (μm)
- D90 (μm)
- s10
- s50
- s90
- a10
- a50
- a90
- FFc (target variable)

## Usage

1. **Data Preparation**:
   - Place your data file in the project directory
   - Update the `file_path` variable in the script with your data file path

2. **Running the Model**:
   ```
   python main.py
   ```
