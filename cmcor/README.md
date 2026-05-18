# Interactive Robotic Moving Cable Segmentation by Motion Correlation

This is the official repository for the letter [Interactive Robotic Moving Cable Segmentation by Motion Correlation](https://doi.org/10.1109/LRA.2025.3574960), IEEE RA-L, 2025 by Ondřej Holešovský, Radoslav Škoviera, Václav Hlaváč.

All the dataset packages are available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15356102.svg)](https://doi.org/10.5281/zenodo.15356102)

```
@Article{Holesovsky2025,
  author    = {Ondrej Holesovsky and Radoslav Skoviera and Vaclav Hlavac},
  journal   = {IEEE Robotics and Automation Letters},
  title     = {{I}nteractive {R}obotic {M}oving {C}able {S}egmentation by {M}otion {C}orrelation},
  year      = {2025},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
}
```


## Table of contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [License](#license)
4. [Code](#code)
    * [Installation](#installation)
    * [Run the code - compute the results](#run-the-code---compute-the-results)
    * [Run the code - analyze the results](#run-the-code---analyze-the-results)
    * [Additional modules](#additional-modules)


## Introduction

![Graphical abstract.](images/graphical-abstract.png)

Manipulating tangled hoses, cables, or ropes can be challenging for both robots and humans. Humans often approach these perceptually demanding tasks by pushing or pulling tangled cables and observing the resulting motions. We follow a similar idea to aid robotic cable manipulation. In this letter, we integrate visual and proprioceptive perception to segment a grasped cable by moving it even when the robot or the grasped cable sometimes perturb neighboring cables. We formulate the cable interactive segmentation problem in such a way that our methods do not require robot arm segmentation masks. Furthermore, a novel grasp sampling method can propose new cable grasp points given a partial cable segmentation to improve the segmentation via additional cable-robot interaction. We evaluate the proposed *motion correlation* (MCor) method on data sequences recorded by our physical robotic setup and show that the method outperforms an earlier *motion segmentation* (MSeg) baseline.

[![Paper video.](videos/paper-video-title.png)](https://data.ciirc.cvut.cz/public/projects/2025CMCor/paper-video.mp4)

(![The same paper video in this repo.](videos/paper-video.mp4))


## Dataset

All the dataset packages are available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15356102.svg)](https://doi.org/10.5281/zenodo.15356102)

We provide the Cable Motion Correlation (CMCor) dataset in a single zip archive.

- [CMCor.zip](https://data.ciirc.cvut.cz/public/projects/2025CMCor/CMCor.zip)
  - size: 42.9 GiB
  - sha256sum: ```231f8887f87a7190b522b9b6f97cc7b46f52fb4f8d2c76d67458c7c10e7b799e```

In addition to the complete dataset package, we provide a sample package with only one recorded (validation) sequence of the dataset:

- [CMCor_sample.zip](https://data.ciirc.cvut.cz/public/projects/2025CMCor/CMCor_sample.zip)
  - size: 447 MiB
  - sha256sum: ```1e6371127e7ed8f8240ef82dd228ee8d792e7729266f6ee209403b16851f1f94```

**Data format**

The dataset files are PNG images and JSON data files. The CMCor archive has two folders:

- ```CMCor/motion_correlation_annotations``` contains binary images of manually created ground truth cable segmentation masks for the last image of each data sequence. Its content has the structure ```dataset_split/sequence_name/cable_mask_DDDDDDDD.png```, where ```dataset_split``` is either ```test``` or ```validation```, ```DDDDDDDD``` is the index of the last image in the sequence, it is zero-padded to eight digits.
- ```CMCor/motion_correlation_buffers``` stores the recorded data sequences. Each sequence contains the following files:
  - ```actions_gripper.json``` - action labels (key ```"action_buffer"```), robot end-effector positions (key ```"ee_point_buffer"```) and other numerical data such as the camera focal length or camera matrix.
  - ```grasped_cable_00000000.png``` - a binary mask image showing the grasped cable segment in the first image of the sequence
  - ```rgb_DDDDDDDD.png``` - color image sequence
  - ```depth_DDDDDDDD.png``` (in all sequences except ```2024-08-06-*```) - depth image sequence, single channel 16-bit PNG images with the depth stored in millimeters
  - ```arm_DDDDDDDD.png``` (not in all sequences) - robot arm binary segmentation mask sequence
  - Corresponding rgb, depth and arm images have the same ```DDDDDDDD``` index. The same index also points to the corresponding action label in ```action_buffer``` and gripper position in ```ee_point_buffer```.

The JSON file ```CMCor/multigrasp_sequences.json``` lists the groups of multigrasp sequences. The sequences in each multigrasp group (the lowest-level list of sequence names in the JSON file) were recorded by grasping and moving the same cable. The first sequence in each group used a grasp given by a human, all the following sequences used automatically proposed grasps.

## License

The CMCor dataset © 2025 by Ondrej Holesovsky, Radoslav Skoviera, Vaclav Hlavac is licensed under CC BY-SA 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/

The source code in this repository is licensed under the MIT license.



## Code

We obtained all the reported runtimes on a desktop computer with an NVIDIA GeForce RTX 2080 Ti and Intel Core i9-9900K CPU @ 3.60GHz.


### Installation

Running the code requires [Python](https://www.python.org) version 3.7 or greater installed on your computer. Furthermore, the Python packages listed in ```requirements.txt``` need to be installed:

```
numpy
matplotlib
opencv-contrib-python   # For cv2. May be also only opencv-python.
scikit-image
scikit-learn
scipy
numpy-quaternion
tqdm
```

Running the code also requires an optical flow estimator. By default, we configured it to use MfnProb FT optical flow estimator from [https://github.com/holesond/movingcables](https://github.com/holesond/movingcables). Please download it and install its requirements if you do not want to use a different optical flow estimator.

To install the packages in a new [virtual environment](https://docs.python.org/3/library/venv.html) at ```/home/user/apps/venv/movingcables```, create and activate the environment first:

```
python -m venv /home/user/apps/venv/movingcables
source /home/user/apps/venv/movingcables/bin/activate
```

To install the requirements, run:

```
pip install -r requirements.txt
```

Activate the environment before each use in a new terminal by:

```
source /home/user/apps/venv/movingcables/bin/activate
```

Next, add the path of the MovingCables flow predictor directory (e.g. ```movingcables/flow_predictors```) to the path search list variable ```folder_list``` in ```cmcor/flow_predictor.py```.

Finally, configure the cmcor software package by changing the variables in ```cmcor/motion_perception_settings.py```:

- ```dataset_root``` - set the path to the CMCor dataset root folder
- ```output_root``` - set the path to an empty folder where to store the computation results (like segmented images etc.)
- ```gpu``` - Leave this set to ```True``` if you want to run optical flow inference on the GPU. Otherwise set this to ```False```.

If you want to change the flow predictor, change the search and import code and/or the class ```FlowPredictor``` in ```cmcor/flow_predictor.py```. The ```FlowPredictor``` class has to have a ```flow``` function which returns an estimated optical flow image given two color images.


### Run the code - compute the results

Compute single grasp motion correlation and segmentation (43m13s):

```
python3 -m cmcor.motion_perception_eval
```


Compute double grasp motion correlation and segmentation (6m11s):

```
python3 -m cmcor.motion_perception_multi_eval
```

(Please note that this second command needs the complete dataset to compute anything.)


### Run the code - analyze the results

The following commands can run only after the results have been computed using the commands above.

Find the optimal method parameters on the validation set (33s):

```
python3 -m cmcor.motion_perception_optimize
```

Show the accuracy of each method (13s):

```
python3 -m cmcor.motion_perception_accuracy
```

Create qualitative plots comparing motion segmentation and correlation methods side-by-side (6m17s):

```
python3 -m cmcor.motion_perception_comparison
```


### Additional modules

- ``cable_pose.py`` - Estimate the cable segment pose in a depth image.
  - Test it using: ``python3 -m cmcor.cable_pose``
- ``pixel_segment.py`` - Return a segment for grasping given a binary mask image and a pixel within the cable mask.
  - Test it using: ``python3 -m cmcor.pixel_segment``
- ``grasp_segment_sampler.py`` - Sample cable segments likely suitable for grasping.
  - Test is using: ``python3 -m cmcor.grasp_segment_sampler``

