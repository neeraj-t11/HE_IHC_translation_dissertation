## Setup
### 1)Envs
- Linux
- Python>=3.6
- CPU or NVIDIA GPU + CUDA CuDNN

Install python packages
```
pip install -r requirements.txt
```
### 2)Prepare dataset
- Download BCI dataset from our homepage.
- Combine HE and IHC images.

  Project provides a python script to generate pix2pix training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene, these can be pairs {HE, IHC}. Then we can learn to translate A(HE images) to B(IHC images).

  Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits.

  Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

  Once the data is formatted this way, call:
  ```
  python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
  ```

  This will combine each pair of images (A,B) into a single image file, ready for training.

- File structure
  ```
  PyramidPix2pix
    ├──datasets
         ├── BCI
               ├──train
               |    ├── 00000_train_1+.png
               |    ├── 00001_train_3+.png
               |    └── ...
               └──test
                    ├── 00000_test_1+.png
                    ├── 00001_test_2+.png
                    └── ...

  ```



### Project Description

This project is part of a master's dissertation titled **"Enhancing Diagnostic Accuracy in H&E to IHC Image Translation: A Study on High HER2 Expression for H&E to IHC Image Translation"**. The goal is to improve diagnostic precision by translating H&E stained slides to IHC images, specifically targeting the identification of high HER2 expression, which is crucial in breast cancer treatment decisions.


The Checkpoints folder contains files pertaining to the training of one experiment. 


The training and testing was done on a gpu node using slurm workload manager. The shell script files are added. 


For the latest updates on the project and access to the complete code, please visit the following repository: https://github.com/neeraj-t11/HE_IHC_translation_dissertation.git
