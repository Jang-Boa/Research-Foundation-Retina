<<<<<<< HEAD
# A Disease-Specific Foundation Model Using Over 100K Fundus Images
=======
## "M-FunFound: A Minimally Supervised Fundus Foundation Model for Multiple Tasks"
>>>>>>> 7bcfc6702972b36bcfebf9cce0e1800ca6d86e8d

**Release and Validation for Abnormality and Multi-Disease Classification on Downstream Tasks**

![overview](image.png)

This repository hosts the official code and pre-trained weights for our disease-specific foundation model for retinal fundus images. The model was pre-trained on over 100,000 fundus images and can be efficiently adapted to a variety of downstream classification tasks.

---

<<<<<<< HEAD
## Key Features
- Pre-trained on ~100K retinal fundus images with supervised learning.
- Two flavors of weights are released:
  - **Fundus** — trained from scratch on fundus images only.
  - **ImageNet + Fundus** — initialized from ImageNet weights, then trained on fundus images.
- Easily fine-tuned or used as a feature extractor for custom downstream tasks (e.g., AMD, Glaucoma, DMR, ERM, RVO, Pathologic myopia).
=======
### Key Features 
- M-FunFound is pre-trained on 0.1 million retinal images with super-vised learning.
- M-FunFound can be efficiently adapted to customised tasks.
>>>>>>> 7bcfc6702972b36bcfebf9cce0e1800ca6d86e8d

---

## Repository Structure
```
Research-Foundation-Retina/
├── Downstream_Task/
│   ├── inference.ipynb        # Example inference notebook
│   └── utils/
│       └── model.py           # Model architecture definition
├── weights/                   # Place downloaded .pth files here (see below)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Pre-trained Weights

The model weights are hosted on Google Drive (file sizes are too large for GitHub):

**Download:** [WEIGHTS](https://drive.google.com/drive/folders/1wMPFan1d5EGaR64FjzMctmxLvXJMoU3D?usp=drive_link)

After downloading, place the `.pth` files inside a `weights/` directory at the project root:

```
weights/
├── resnet50_512_Fundus.pth
└── resnet50_512_ImageNet+Fundus.pth
```

| File | Backbone | Input size | Pre-training |
|---|---|---|---|
| `resnet50_512_Fundus.pth` | ResNet-50 | 512×512 | Fundus only |
| `resnet50_512_ImageNet+Fundus.pth` | ResNet-50 | 512×512 | ImageNet → Fundus |

---

## Installation

We recommend Python 3.9+ and a CUDA-capable GPU.

```bash
git clone https://github.com/Jang-Boa/Research-Foundation-Retina.git
cd Research-Foundation-Retina
pip install -r requirements.txt
```

Tested with PyTorch 2.0.1 + CUDA.

---

## Usage

### Inference
Open and run the example notebook:

```bash
jupyter notebook Downstream_Task/inference.ipynb
```

The notebook demonstrates how to load the pre-trained weights and run inference on a fundus image.

### Fine-tuning on a Custom Dataset
Use `Downstream_Task/utils/model.py` to instantiate the architecture, then load the pre-trained checkpoint as initialization for your own training loop:

```python
import torch
from Downstream_Task.utils.model import ResNetModel

model = ResNetModel("resnet50", num_classes=<YOUR_NUM_CLASSES>)
state_dict = torch.load("weights/resnet50_512_ImageNet+Fundus.pth", map_location="cpu")
# Then fine-tune on your dataset.
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{<TBD>,
  title   = {A Disease-Specific Foundation Model Using Over 100K Fundus Images: Release and Validation for Abnormality and Multi-Disease Classification on Downstream Tasks},
  author  = {<TBD>},
  journal = {<TBD>},
  year    = {<TBD>}
}
```

---

## License

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. See [LICENSE](LICENSE) for the full text.

The pre-trained weights are released under the same license and are intended for **non-commercial research use only**.

---

## Contact

For questions, please reach out:
- **balinda@snu.ac.kr**
- **boajang97@gmail.com**

---

## Acknowledgements

We thank all collaborators and institutions that contributed fundus images and annotations used in the development of this foundation model.
