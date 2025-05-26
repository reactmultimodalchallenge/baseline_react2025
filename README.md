# Official baseline code for the Third REACT Challenge (REACT2025)
[[Homepage]](https://sites.google.com/view/react2025)  [[Reference Paper (TBA)]]() [[Code]](https://github.com/reactmultimodalchallenge/baseline_react2025)

This repository provides baseline methods for the [Third REACT Challenge](https://sites.google.com/cam.ac.uk/react2024)

### Baseline paper:
- https://arxiv.org/pdf/2505.17223

### MARS dataset:
- Please send the signed EULA (https://github.com/reactmultimodalchallenge/baseline_react2025/blob/main/EULA_MARS%20dataset.pdf) to Dr Siyang Song at s.song@exeter.ac.uk 

### Challenge Description
Given the spatio-temporal behaviours expressed by a speaker at the time period, the proposed REACT 2025 Challenge will consist of the following two sub-challenges whose theoretical underpinnings have been defined and detailed in this paper.

#### Task 1 - Offline Appropriate Facial Reaction Generation
This task aims to develop a deep learning model that takes the entire speaker behaviour sequence as the input, and generates multiple appropriate and realistic / naturalistic spatio-temporal facial reactions, consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 
#### Task 2 - Online Appropriate Facial Reaction Generation
This task aims to develop a deep learning model that estimates each frame, rather than taking all frames into consideration. The model is expected to gradually generate all facial reaction frames to form multiple appropriate and realistic / naturalistic spatio-temporal facial reactions consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 

[//]: # (https://github.com/reactmultimodalchallenge/baseline_react2023/assets/35754447/8c7e7f92-d991-4741-80ec-a5112532460b)


## 🛠️ Dependency Installation

We provide detailed instructions for setting up the environment using conda. First, create and activate a new environment:
``` shell
conda create -n react python=3.10
conda activate react
```

### 1. Install PyTorch
First, check your CUDA version:
``` shell
nvidia-smi
```
Visit [Pytorch official website](https://pytorch.org/) to get the appropriate installation command. For example:
``` shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install PyTorch3D Dependencies
Install the following dependencies:
``` shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For CUDA versions older than 11.7, you will need to install the CUB library. 
``` shell
conda install -c bottler nvidiacub
```

### 3. Install PyTorch3D
First, verify your CUDA version in Python:
``` shell
import torch
torch.version.cuda
```
[//]: # (Download `pytorch3d` file based on the version of python, cuda and pytorch from https://anaconda.org/pytorch3d/pytorch3d/files. For example, to install for Python 3.8, PyTorch 1.12.1 and CUDA 11.6, select the below file to download)
Download the appropriate `PyTorch3D` package from [Anaconda](https://anaconda.org/pytorch3d/pytorch3d/files) based on your Python, CUDA, and PyTorch versions. For example, for Python 3.10, CUDA 11.6, and PyTorch 1.12.0:

[//]: # (Finally install `pytorch3d` via the downloaded `.tar.bz2` file via conda)
``` shell
# linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
conda install linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
```

### 4. Install Additional Dependencies
[//]: # (pip install omegaconf scikit-video pandas soundfile av decord tensorboard numpy tslearn scikit-image matplotlib imageio plotly opencv-python librosa einops)
Install all remaining dependencies specified in requirements.txt:
``` shell
pip install -r requirements.txt
```


## 👨‍🏫 Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description ([Homepage](https://sites.google.com/cam.ac.uk/react2024)):**

We divided the datasets into training, test, and validation sets following an estimated 60%/20%/20% splitting ratio. Specifically, we split the datasets with a subject-independent strategy (i.e., the same subject was never included in the train and test sets).

[//]: # (- Dataset Directory Structure: &#40;training and validation sets are provided at this stage&#41;)
- *video-raw* folder contains raw videos (with the resolution of 1920 * 1080)
- *video-face-crop* folder contains face-cropped videos (with the resolution of 384 * 384)
- *facial-attributes* folder contains sequences of frame-level 25-dimension facial attributes (15 AUs’ occurrences, valence and arousal intensities, and the probabilities of eight categorical facial expressions)
- *coefficients* folder contains sequences of 58-dimension (52-d expression, 3-d rotation, and 3-d translation) 3DMM coefficients extracted from corresponding videos
- *audio* folder contains wav files extracted from raw video files

Appropriate real facial reactions (Ground-Truths):
- During data recording, the semantic contexts are carefully controlled through the 23 distinct sessions (session0, session1, …, session22), each of which is guided by a few pre-defined sentences posted by the speaker. This provides a consistent session-specific context across dyadic interactions between different speakers and listeners. More specifically, for the speaker behaviour expressed in a specific session, we define all facial reactions expressed by different listeners under the same session to be appropriate facial reactions (i.e., ground-truth) for responding to it.
   
**Data organization (`./data`) is listed below:**
The example of data structure.
```

├── val
├── test
├── train
    ├── coefficients (.npy)
    ├── video-face-crop (.mp4)
    ├── video-raw (.mp4)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
    ├── facial-attributes (.npy)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
    ├── audio (.wav)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
```

</p>
</details>

<details><summary> <b> External Tool Preparation </b> </summary>
<p>

We use 3DMM coefficients to represent a 3D listener or speaker, and for further 3D-to-2D frame rendering. The baselines leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients, and render 3D facial reactions.  

- You should first download 3DMM (FaceVerse version 2 model) at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and then put it in the folder (`external/FaceVerse/data/`).
 
  We provide our extracted 3DMM coefficients (which are used for our baseline visualisation) at [OneDrive](https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu). 

  We also provide the `mean_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EXhSObCHXUhHg0-Geyy4_6QB7b611XFgbJcIoGymcmzS-Q?e=NT8IKj) and `std_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EdyIBxX-IlVEivdFxURn-BMBiK6JFSAXcp3qwCPNVboifQ?e=o5NgqM) and `reference_full.npy` at this [Onedrive link](https://1drv.ms/u/c/4c787027becb2e91/ERoBr5MNudxBgImW4jPt39sBwqFNSsvwX3OihUfU_TYpqw?e=h8mOqp) for 3DMM coefficients Data Normalization. Please download and put them in the folder (`external/FaceVerse/`).

[//]: # ( and reference_full )

Then, we use a 3D-to-2D tool [PIRender](https://github.com/RenYurui/PIRender) to render final 2D facial reaction frames.
 
- We re-trained the PIRender, and the well-trained model is provided at the [checkpoint](https://1drv.ms/u/c/4c787027becb2e91/ERLUL_QTBABHoLzCTCbUZF8Bu6e_5o0YX31rA12yv0DIcQ?e=mWKgcn). Please put it in the folder (`external/PIRender/`).

Finally, please download the compressed folder named `pretrained_models` from [this link](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb), and extract it into the project root directory.

</p>
</details>


<details><summary> <b> Training </b>  </summary>
<p>
 
 <b>Trans-VAE</b>
- Running the following shell can start training Trans-VAE baseline for the offline task:
 ```shell
 python main.py \
    data=motion_transvae \
    trainer=motion_transvae \
    trainer.batch_size=4 \
    trainer.max_seq_len=750 \
    trainer.window_size=8 \
    stage=fit \
    task=offline \
    data_dir=./data
 ```
 &nbsp; &nbsp; or for the online task:
 
  ```shell
 python main.py \
    data=motion_transvae \
    trainer=motion_transvae \
    trainer.batch_size=2 \
    trainer.max_seq_len=256 \
    trainer.window_size=16 \
    stage=fit \
    task=online \
    data_dir=./data
 ```
 
 <b>PerFRDiff</b>
 - Running the following shell can start training PerFRDiff baseline for the offline task: 
```shell
python main.py \
    data=motion_diffusion \
    trainer=motion_diffusion \
    trainer.batch_size=2 \
    stage=fit \
    task=offline \
    data_dir=./data
```
 &nbsp; &nbsp; or for the online task:
```shell
python main.py \
    data=motion_diffusion \
    trainer=motion_diffusion \
    trainer.batch_size=8 \
    stage=fit \
    task=online \
    data_dir=./data
```

 <b>REGNN</b>
 - Make sure you are in the folder `regnn` before running any cells related to REGNN.
 - First extract the image features using the pre-trained swin_transformer (pretrained weights already provided in `pretrained_models`).
 ```shell
 python feature_extraction.py
 ```

 - Then train the REGNN by running the following shell:
 ```shell
 python train.py \
     --logs-dir='Gmm-logs' \
     --milestones=9 \
     --batch-size=64 \
     --layers=2 \
     --norm \
     --neighbor-pattern='all' \
     --convert-type='direct' \
     --loss-mid \
     --data-dir=../data
 ```
 
</p>
</details>

<details><summary> <b> Pretrained weights </b>  </summary>

- [ ] to be released

</details>

<details><summary> <b> Evaluation </b>  </summary>

- [ ] to be released

</details>


## 🖊️ Citation

### Submissions should cite the following papers:

#### Theory paper and baseline paper:

[1] Song, Siyang, Micol Spitale, Yiming Luo, Batuhan Bal, and Hatice Gunes. "Multiple Appropriate Facial Reaction Generation in Dyadic Interaction Settings: What, Why and How?." arXiv preprint arXiv:2302.06514 (2023).

[2] Song, Siyang, Micol Spitale, Cheng Luo, Cristina Palmero, German Barquero, Hengde Zhu, Sergio Escalera et al. "REACT 2024: the Second Multiple Appropriate Facial Reaction Generation Challenge." arXiv preprint arXiv:2401.05166 (2024).

[3] Song, Siyang, Micol Spitale, Cheng Luo, Germán Barquero, Cristina Palmero, Sergio Escalera, Michel Valstar et al. "REACT2023: The First Multiple Appropriate Facial Reaction Generation Challenge." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9620-9624. 2023.

#### Annotation, basic feature extraction tools and baselines:

[6] Song, Siyang, Yuxin Song, Cheng Luo, Zhiyuan Song, Selim Kuzucu, Xi Jia, Zhijiang Guo, Weicheng Xie, Linlin Shen, and Hatice Gunes. "GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features." arXiv preprint arXiv:2211.12482 (2022).

[7] Luo, Cheng, Siyang Song, Weicheng Xie, Linlin Shen, and Hatice Gunes. (2022, July) "Learning multi-dimensional edge feature-based au relation graph for facial action unit recognition." Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (pp. 1239-1246).

[8] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "Estimation of continuous valence and arousal levels from faces in naturalistic conditions." Nature Machine Intelligence 3, no. 1 (2021): 42-50.

[9] Eyben, Florian, Martin Wöllmer, and Björn Schuller. "Opensmile: the munich versatile and fast open-source audio feature extractor." In Proceedings of the 18th ACM international conference on Multimedia, pp. 1459-1462. 2010.

### Submissions are encouraged to cite previous facial reaction generation papers:

[1] Huang, Yuchi, and Saad M. Khan. "Dyadgan: Generating facial expressions in dyadic interactions." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 11-18. 2017.

[2] Huang, Yuchi, and Saad Khan. "A generative approach for dynamically varying photorealistic facial expressions in human-agent interactions." In Proceedings of the 20th ACM International Conference on Multimodal Interaction, pp. 437-445. 2018.

[3] Shao, Zilong, Siyang Song, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Personality recognition by modelling person-specific cognitive processes using graph representation." In proceedings of the 29th ACM international conference on multimedia, pp. 357-366. 2021.

[4] Song, Siyang, Zilong Shao, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Learning Person-specific Cognition from Facial Reactions for Automatic Personality Recognition." IEEE Transactions on Affective Computing (2022).

[5] Barquero, German, Sergio Escalera, and Cristina Palmero. "Belfusion: Latent diffusion for behavior-driven human motion prediction." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2317-2327. 2023.

[6] Zhou, Mohan, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. "Responsive listening head generation: a benchmark dataset and baseline." In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII, pp. 124-142. Cham: Springer Nature Switzerland, 2022.

[7] Luo, Cheng, Siyang Song, Weicheng Xie, Micol Spitale, Linlin Shen, and Hatice Gunes. "ReactFace: Multiple Appropriate Facial Reaction Generation in Dyadic Interactions." arXiv preprint arXiv:2305.15748 (2023).

[8] Xu, Tong, Micol Spitale, Hao Tang, Lu Liu, Hatice Gunes, and Siyang Song. "Reversible Graph Neural Network-based Reaction Distribution Learning for Multiple Appropriate Facial Reactions Generation." arXiv preprint arXiv:2305.15270 (2023).

[9] Liang, Cong, Jiahe Wang, Haofan Zhang, Bing Tang, Junshan Huang, Shangfei Wang, and Xiaoping Chen. "Unifarn: Unified transformer for facial reaction generation." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9506-9510. 2023.

[10] Yu, Jun, Ji Zhao, Guochen Xie, Fengxin Chen, Ye Yu, Liang Peng, Minglei Li, and Zonghong Dai. "Leveraging the latent diffusion models for offline facial multiple appropriate reactions generation." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9561-9565. 2023.

[11] Hoque, Ximi, Adamay Mann, Gulshan Sharma, and Abhinav Dhall. "BEAMER: Behavioral Encoder to Generate Multiple Appropriate Facial Reactions." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9536-9540. 2023.

[12] Zhu, Hengde, Xiangyu Kong, Weicheng Xie, Xin Huang, Linlin Shen, Lu Liu, Hatice Gunes, and Siyang Song. "Perfrdiff: Personalised weight editing for multiple appropriate facial reaction generation." In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 9495-9504. 2024.

[13] Zhu, Hengde, Xiangyu Kong, Weicheng Xie, Xin Huang, Xilin He, Lu Liu, Linlin Shen, Wei Zhang, Hatice Gunes, and Siyang Song. "PerReactor: Offline Personalised Multiple Appropriate Facial Reaction Generation." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 2, pp. 1665-1673. 2025.
  

## 🤝 Acknowledgement
Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) &#8194;

- [PIRender](https://github.com/RenYurui/PIRender) &#8194;

[//]: # (<details><summary> <b> Validation </b>  </summary>)

[//]: # (<p>)

[//]: # ( Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.)

[//]: # ( )
[//]: # (- Before validation, run the following script to get the martix &#40;defining appropriate neighbours in val set&#41;:)

[//]: # ( ```shell)

[//]: # ( cd tool)

[//]: # ( python matrix_split.py --dataset-path ./data --partition val)

[//]: # ( ```)

[//]: # (&nbsp;  Please put files &#40;`data_indices.csv`, `Approprirate_facial_reaction.npy` and `val.csv`&#41; in the folder `./data/`.)

[//]: # (  )
[//]: # (- Then, evaluate a trained model on val set and run:)

[//]: # ()
[//]: # ( ```shell)

[//]: # (python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/val_offline --split val)

[//]: # (```)

[//]: # ( )
[//]: # (&nbsp; or)

[//]: # ( )
[//]: # (```shell)

[//]: # (python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/val_online --split val)

[//]: # (```)

[//]: # ( )
[//]: # (- For computing FID &#40;FRRea&#41;, run the following script:)

[//]: # ()
[//]: # (```)

[//]: # (python -m pytorch_fid  ./results/val_offline/fid/real  ./results/val_offline/fid/fake)

[//]: # (```)

[//]: # (</p>)

[//]: # (</details>)


[//]: # (<details><summary> <b> Other baselines </b>  </summary>)

[//]: # (<p>)

[//]: # ( )
[//]: # (- Run the following script to sequentially evaluate the naive baselines presented in the paper:)

[//]: # ( ```shell)

[//]: # ( python run_baselines.py --split SPLIT)

[//]: # ( ```)

[//]: # ( SPLIT can be `val` or `test`.)

[//]: # (</p>)

[//]: # (</details>)


[//]: # (<details><summary> <b> Pretrained weights </b>  </summary>)

[//]: # ( If you would rather skip training, download the following checkpoints and put them inside the folder './results'.)

[//]: # (<p>)

[//]: # ( )
[//]: # ( <b>Trans-VAE</b>: TBA)

[//]: # ( )
[//]: # ( <b>BeLFusion</b>: [download]&#40;https://ubarcelona-my.sharepoint.com/:f:/g/personal/germanbarquero_ub_edu/EkRisY7MzX5MnP6tIVYhkdYBInl3lw3XXJuW6fEXnij4aQ?e=XZHvSw&#41;)

[//]: # ()
[//]: # ( <b>REGNN</b>: [download]&#40;https://drive.google.com/drive/folders/18I-yfpY1mlLqp4-E443xxwXNWh3ET-RN?usp=sharing&#41;)

[//]: # ( )
[//]: # (</details>)

[//]: # ()
[//]: # (<details><summary> <b> Evaluation </b>  </summary>)

[//]: # (<p>)

[//]: # ( Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.)

[//]: # ( )
[//]: # (- Before testing, run the following script to get the martix &#40;defining appropriate neighbours in test set&#41;:)

[//]: # ( ```shell)

[//]: # ( cd tool)

[//]: # ( python matrix_split.py --dataset-path ./data --partition test)

[//]: # ( ```)

[//]: # (&nbsp;  Please put files &#40;`data_indices.csv`, `Approprirate_facial_reaction.npy` and `test.csv`&#41; in the folder `./data/`.)

[//]: # (  )
[//]: # (- Then, evaluate a trained model on test set and run:)

[//]: # ()
[//]: # ( ```shell)

[//]: # (python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/test_offline --split test)

[//]: # (```)

[//]: # ( )
[//]: # (&nbsp; or)

[//]: # ( )
[//]: # (```shell)

[//]: # (python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/test_online --split test)

[//]: # (```)

[//]: # ()
[//]: # ( )
[//]: # (- For computing FID &#40;FRRea&#41;, run the following script:)

[//]: # ()
[//]: # (```)

[//]: # (python -m pytorch_fid  ./results/test_offline/fid/real  ./results/test_offline/fid/fake)

[//]: # (```)

[//]: # ()
[//]: # ( For evaluation of REGNN, there are two steps.)

[//]: # ( - First generate facial reactions and save them by running the script within the folder `regnn`:)

[//]: # ( ```)

[//]: # ( bash scripts/inference.sh)

[//]: # ( ```)

[//]: # ( - Then evaluate the predicted facial reactions by running the `evaluation.py` in the folder `regnn`:)

[//]: # ( ```)

[//]: # ( python evaluation.py --data-dir <data-dir> --pred-dir <pred-dir> split test)

[//]: # ( ```)

[//]: # (</p>)

[//]: # (</details>)
