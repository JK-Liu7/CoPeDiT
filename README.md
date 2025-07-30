# CoPeDiT
Code for the paper "Exploiting Completeness Perception with Diffusion Transformer for Unified 3D MRI Synthesis"

## Abstract
Missing data problems, such as missing modalities in multi-modal brain MRI and missing slices in cardiac MRI, pose significant challenges in clinical practice. Existing methods rely on external guidance to supply detailed missing state for instructing generative models to synthesize missing MRIs. However, manual indicators are not always available or reliable in real-world scenarios due to the unpredictable nature of clinical environments. Moreover, these explicit masks are not informative enough to provide guidance for improving semantic consistency. In this work, we argue that generative models should infer and recognize missing states in a self-perceptive manner, enabling them to better capture subtle anatomical and pathological variations. Towards this goal, we propose CoPeDiT, a general-purpose latent diffusion model equipped with ***completeness perception*** for unified synthesis of 3D MRIs. Specifically, we incorporate dedicated pretext tasks into our tokenizer, CoPeVAE, empowering it to learn completeness-aware discriminative prompts, and design MDiT3D, a specialized diffusion transformer architecture for 3D MRI synthesis, that effectively uses the learned prompts as guidance to enhance semantic consistency in 3D space. Comprehensive evaluations on three large-scale MRI datasets demonstrate that CoPeDiT significantly outperforms state-of-the-art methods, achieving superior robustness, generalizability, and flexibility.

![teaser](assets/CoPeVAE.png)

![teaser](assets/MDiT3D.png)

## Prepare Dataset
First, you need to download the brain and cardiac MRI datasets. All dataset used in our experiment are open-source except UKBB and MESA, and you can download yourself.

![teaser](assets/Dataset.png)

BraTS 2021: https://www.synapse.org/#!Synapse:syn25829067/wiki/610863 

IXI: https://brain-development.org/ixi-dataset/ 

ACDC: https://www.creatis.insa-lyon.fr/Challenge/acdc/

MSCMR: https://zmiclab.github.io/zxh/0/mscmrseg19/data.html

The structure of our dataset folder is:
```
├── dataset
    ├── BrainMRI
      ├── BraTS2021
      └── IXI
    ├── CardiacMRI
      ├── UKBB
      ├── MESA
      ├── ACDC
      └── MSCMR
```

## Usage


## Acknowledgement
Our code is implemented based on [MONAI](https://github.com/Project-MONAI/research-contributions), We thank for part of their codes.
