# Generative Earthquake Modelling
During my Master's Degree in Data Science with Artificial Intelligence at The University of Exeter, I undertook a research project which involved the development and implementation of a deep learning framework for generating synthetic earthquake response spectra from an interpretable and hierarchical latent space. The final report can be found in the repository, `report.pdf` with detail on the project.
## Contents
The repository is structured with a series of `.py` files containing the classes and functions created for the project, a Jupyter notebook (`validation.ipynb`) with an example training and validation workflow of the proposed model, and a folder (`FINAL RESULTS`) containing the final results that were used during the final report. These are the programme files and their contents:
* `data.py`: Functions relating to the reading and pre-processing of the data.
* `poincare.py`: Functions relating to Poincaré ball geometry which were used for the latent space of the proposed model. The class definition of a Gyroplane layer is also included.
* `p_h_vae.py`: Class definition for the Poincaré ball hierarchical variational autoencoder, the proposed model for this project. A function that was used to generate the encoder and decoder neural networks is also included.
* `nn.py`: Class definition for the two-stage neural network used to estimate feature values for points in the latent space.
* `ffnn.py`: Class definition for the feed-forward neural network used to map live intensity measures to points in the latent space for early warning applications.
* `validation.ipynb`: Jupyter notebook with a training and validation example of the proposed model.
## Input Dataset
The input dataset for this project was the NGA West2 ground motion dataset (https://ngawest2.berkeley.edu/). The proposed model is trained using the acceleration response spectra in `Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx`. The FFNN for early warning is trained using the intensity measures in `First3secs_IMs.csv`, where the field ```RSN``` maps to ```Record Sequence Number``` in the original dataset.
## Instructions
The `.py` files in this repository can be imported for use and have no function when ran independently. `validation.ipynb` can be ran, assuming the `.py` files are saved in the same directory for import, and user instructions are provided within. In summary, to run `validation.ipynb`, the user must specify filepaths to the two input datasets and a sub-folder directory to save model outputs in (or read previous model outputs from). If ```retrain=True```, then a new instance of the model will be trained and saved in ```data_dir```, otherwise previous model outputs will be read from ```data_dir```. All validation found in the final report is subsequently performed with the trained model.
## Publication
Following the research project submission, I completed further work and co-authored a paper called *Structured Generative Modelling of Earthquake Response Spectra with Hierarchical Latent Variables in Hyperbolic Geometry* published in **Scientific Reports** - https://doi.org/10.1038/s41598-025-29902-6.
### *Requirements*
*For this project, I used Python version 3.10.6 and TensorFlow version 2.16.1.*
