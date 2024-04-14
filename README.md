# TerrainGen
This is a Streamlit web app that generates Heightmaps based off of  user provided sketches using a custom Generator Model I've built and Trained using Tensorflow.

The Generator was trained using Two critics, with the generator essentially being shared by two GAN Models, namely
- Conversion GAN
- Realism GAN

Of these two critics, The Conversion critic was used to evaluate and train the Generator on how well it replicates the user's input sketch. And the Realism critic was used to evaluate the Generator based on the quality of its produced Heightmaps.

This project was mostly implemented by following the paper [[1]](#1). 

<hr>

## Dataset
The Dataset used was created using data from : https://earthexplorer.usgs.gov/ (3 arc second  resolution images from the SRTM void-filled dataset)
The data used mainly consisted of terrain that covers parts of Europe and India. 
Data Augmentation was done following the methods mentioned in [[1]](#1). The resulting images were downscaled to 128x128 resolution due to hardware limitations. To account for this the resolution of the images extracted by the sliding window and the offset of the sliding window itself are different compared to what was mentioned in [[1]](#1). As a result, the images used for training and the results produced by the generator would conver less surface area, compared to the results observed in [[1]](#1).

<hr>

## Training
The training was done by alternating between training the generator with the conversion critic and the realism critic. The criterium used to select which GAN to trainat any given point is the following : 
- if loss of conversion critic with fake data > 2 * loss of realism critic with fake data, select the conversion critic, else select the realism critic.

The generator and the criitcs all use **Wasserstein loss** as their loss function.
<br>
<hr>

## References
<a id="1">[1]</a>
Nuno Ramos, Pedro A. Santos, Joao Dias. [(2023)](https://doi.org/10.1145/3582437.3587183), 
“Dual Critic Conditional Wassertein GAN for Height-Map Generation”, 
FDG’23, Article No.: 45, pp. 1-4.
