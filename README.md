# GAN semantics for personalized face beauty synthesis and beautification

The framework generate the face based on personal beauty preferences of an individual. MEBeauty dataset is used for this purpose.
The personalized face beautification is also implemented.

## Usage

### 1. Clone these repos.
```bash
git clone https://github.com/irina-lebedeva/MEBeautydatabase
git clone https://github.com/irina-lebedeva/GAN-ideal-face
cd GAN-ideal-face/
```

### 2. Crop and align faces.

```bash
 python face_crop_align.py  --images_path [path to the folder with images] 
 --results_path [folder where the cropped images should be saved] 
 --method [one of the backends mentioned above]
    
```
Default folder  - `/images/`
Default backend `opencv`

Crop and alignment is a required step for correct GAN encoding

### 3. Create a list with top m images in n training set for all rater

```bash
selected_images.ipynb   
```
Adjust the number of training samples and number of top rated images

### 4. Genarate the ideal face

```bash
 python ideal.py  --file [csv file with top rated images by users]  
                  --folder [folder where dave the ideal faces of raters]
```

### 5. Beautify a face image based on an individual's personal preferences *

```bash
 python beautify.py  --image [face image for beautification]  
                     --folder [the rated ID]
```
* only after successfully conducted step 4
* 
Based on **In-Domain GAN Inversion for Real Image Editing**

[[Paper](https://arxiv.org/pdf/2004.00049.pdf)] [[Official Code](https://github.com/genforce/idinvert)] [[StyleGAN2 Pytorch](https://github.com/rosinality/stylegan2-pytorch)]






