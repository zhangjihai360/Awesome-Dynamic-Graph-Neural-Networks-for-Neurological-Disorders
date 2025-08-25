# Awesome Dynamic Graph Neural Networks for Neurological Disorders

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of resources for Dynamic Graph Neural Networks (DGNNs) in neurological disorder modeling and recognition, including papers, datasets, and implementations.

## 📋 Table of Contents
- [Overview](#overview)
- [Papers by Disease/Application](#papers-by-diseaseapplication)
  - [Epilepsy](#epilepsy)
  - [Alzheimer's Disease](#alzheimers-disease)
  - [Autism Spectrum Disorder](#autism-spectrum-disorder)
  - [Depression](#depression)
  - [Cognitive Impairment](#cognitive-impairment)
  - [Emotion Recognition](#emotion-recognition)
  - [Brain Modeling](#brain-modeling)
- [Papers by Architecture](#papers-by-architecture)
  - [RNN-based Models](#rnn-based-models)
  - [TCN-based Models](#tcn-based-models)
  - [Attention/Transformer-based Models](#attentiontransformer-based-models)
- [Datasets](#datasets)
- [Tools and Resources](#tools-and-resources)
- [Contributing](#contributing)

## Overview

This repository maintains a comprehensive collection of research on Dynamic Graph Neural Networks (DGNNs) applied to neurological disorders. DGNNs have emerged as powerful tools for modeling the complex spatiotemporal dynamics of brain networks, offering significant advantages over traditional static approaches in capturing disease progression and neural activity patterns.

### Key Features of DGNNs in Neurological Applications:
- **Temporal Dynamics Modeling**: Capture time-evolving brain network patterns
- **Multi-modal Integration**: Combine EEG, fMRI, DTI and other neuroimaging data
- **Disease-specific Patterns**: Identify unique spatiotemporal signatures of neurological conditions
- **Interpretability**: Provide insights into disease mechanisms through attention mechanisms

## Papers by Disease/Application

### Epilepsy

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2025 | [Automated seizure detection using dynamic temporal-spatial graph attention network](https://doi.org/10.1038/s41598-025-01015-0) | Attention-based | EEG | - |
| 2024 | [Dynamic GNNs for precise seizure detection and classification from EEG data](http://arxiv.org/abs/2405.09568) | RNN-based (GC-GRU) | EEG | - |
| 2024 | [Time-series anomaly detection based on dynamic temporal graph convolutional network](https://doi.org/10.3390/bioengineering11010053) | TCN-based (DTGCN) | EEG | - |
| 2022 | [Graph-generative neural network for EEG-based epileptic seizure detection](https://doi.org/10.1038/s41598-022-23656-1) | RNN-based (GGN) | EEG | [Code](https://github.com/) |
| 2022 | [Self-supervised graph neural networks for improved electroencephalographic seizure analysis](https://doi.org/10.48550/arXiv.2104.08336) | RNN-based (GC-GRU) | EEG | - |
| 2019 | [Temporal graph convolutional networks for automatic seizure detection](https://doi.org/10.48550/arXiv.1905.01375) | TCN-based | EEG | - |

### Alzheimer's Disease

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2024 | [Predictive modeling of alzheimer's disease progression using temporal clinical factors](https://doi.org/10.1016/j.ibmed.2024.100159) | TCN+GAT | Multimodal | - |
| 2024 | [Interpretable spatio-temporal embedding for brain structural-effective network with ODE](https://doi.org/10.1007/978-3-031-72069-7_22) | ODE-based | Multimodal | - |
| 2021 | [DS-GCNs: Connectome classification using dynamic spectral graph convolution networks](https://doi.org/10.1093/cercor/bhaa292) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2020 | [Dynamic functional connectivity and graph convolution network for AD classification](https://doi.org/10.48550/arXiv.2006.13510) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2020 | [Spatial-temporal dependency modeling and network hub detection for functional MRI](https://doi.org/10.1109/TBME.2019.2957921) | RNN-based (GC-LSTM) | rs-fMRI | - |

### Autism Spectrum Disorder

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2025 | [Diagnosis of ASD by dynamic functional connectivity using GNN-LSTM](https://doi.org/10.3390/s25010156) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2024 | [Dual-view connectivity analysis via dynamic graph transformer network for ASD](https://doi.org/10.1109/ISBI56570.2024.10635496) | Transformer-based | rs-fMRI | - |
| 2023 | [BrainTGL: A dynamic graph representation learning model for brain network analysis](https://doi.org/10.1016/j.compbiomed.2022.106521) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2022 | [Spatio-temporal attention in multi-granular brain chronnectomes for ASD detection](https://doi.org/10.48550/arXiv.2211.07360) | Transformer-based | rs-fMRI | - |

### Depression

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2026 | [A graph transformer-based foundation model for brain functional connectivity network](https://doi.org/10.1016/j.patcog.2025.111988) | Transformer-based | rs-fMRI | - |
| 2025 | [Using dynamic graph convolutional network to identify individuals with MDD](https://doi.org/10.1016/j.jad.2024.11.035) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2024 | [Spatio-temporal learning and explaining for dynamic functional connectivity in depression](https://doi.org/10.1016/j.jad.2024.08.014) | Transformer-based | rs-fMRI | - |
| 2021 | [Spatio-temporal graph convolutional network for MDD diagnosis and treatment prediction](https://doi.org/10.1002/hbm.25529) | TCN-based (STGCN) | rs-fMRI | - |

### Cognitive Impairment

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2025 | [Dynamically weighted graph neural network for detection of early MCI](https://doi.org/10.1371/journal.pone.0323894) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2024 | [Leveraging brain modularity prior for interpretable representation learning of fMRI](https://doi.org/10.1109/TBME.2024.3370415) | Attention-based | rs-fMRI | - |
| 2023 | [FE-STGNN: Spatio-temporal graph neural network with FC and EC fusion for MCI diagnosis](https://doi.org/10.1007/978-3-031-43993-3_7) | Attention-based | rs-fMRI | - |
| 2021 | [Building dynamic hierarchical brain networks for early MCI diagnosis](https://doi.org/10.1007/978-3-030-87234-2_54) | RNN-based (GC-LSTM) | rs-fMRI | - |
| 2019 | [Dynamic spectral graph convolution networks with assistant task training for early MCI](https://doi.org/10.1007/978-3-030-32251-9_70) | RNN-based (GC-LSTM) | fMRI | - |

### Emotion Recognition

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2025 | [Spatio-temporal graph BERT network for EEG emotion recognition](https://doi.org/10.1016/j.bspc.2025.107576) | Transformer-based | EEG | - |
| 2025 | [DGAT: A dynamic graph attention neural network framework for EEG emotion recognition](https://doi.org/10.3389/fpsyt.2025.1633860) | Attention-based | EEG | - |
| 2025 | [A spatio-temporal graph neural network for EEG emotion recognition](https://doi.org/10.2298/csis250215053w) | RNN-based (GC-GRU) | EEG | - |
| 2024 | [ST-SCGNN: Spatio-temporal self-constructing GNN for cross-subject EEG emotion recognition](https://doi.org/10.1109/jbhi.2023.3335854) | TCN-based | EEG | - |
| 2023 | [A domain generative graph network for EEG-based emotion recognition](https://doi.org/10.1109/JBHI.2023.3242090) | RNN-based (GC-LSTM) | EEG | [Code](https://github.com/) |
| 2023 | [STGATE: Spatial-temporal graph attention network with transformer encoder](https://doi.org/10.3389/fnhum.2023.1169949) | Transformer-based | EEG | - |
| 2020 | [EEG emotion recognition using dynamical graph convolutional neural networks](https://doi.org/10.1109/TAFFC.2018.2817622) | RNN-based (GC-LSTM) | EEG | - |

### Brain Modeling

| Year | Paper | Model Type | Data Type | Code |
|------|-------|------------|-----------|------|
| 2022 | [Revealing continuous brain dynamical organization with multimodal graph transformer](https://doi.org/10.1007/978-3-031-16431-6_33) | Transformer-based | Multimodal | - |
| 2021 | [Learning dynamic graph representation of brain connectome with spatio-temporal attention](https://doi.org/10.48550/arXiv.2105.13495) | Attention-based (STAGIN) | fMRI | [Code](https://github.com/) |
| 2021 | [Spatio-temporal graph convolution for resting-state fMRI analysis](https://doi.org/10.48550/arXiv.2003.10613) | TCN-based | rs-fMRI | - |

## Papers by Architecture

### RNN-based Models

#### GC-LSTM (Graph Convolutional LSTM)
- Combines graph convolution with LSTM gates for tight coupling of spatial and temporal features
- Best for: Long-term dependency modeling, sequential pattern recognition

#### GC-GRU (Graph Convolutional GRU)
- Modular design with separate spatial and temporal components
- Best for: Flexible architecture, easier training

### TCN-based Models

#### ST-GCN (Spatio-Temporal Graph Convolutional Network)
- Uses 1D causal convolution for temporal modeling
- Best for: Parallel processing, stable gradient propagation

#### With Dilated Convolution
- Extended receptive field for long-range dependencies
- Best for: Multi-scale temporal pattern extraction

### Attention/Transformer-based Models

#### Graph Attention Networks
- Dynamic weight allocation across spatial and temporal dimensions
- Best for: Interpretability, identifying important brain regions

#### Graph Transformers
- Global dependency modeling with multi-head attention
- Best for: Complex interaction patterns, large-scale networks

## Datasets

### EEG Datasets

| Dataset | Size | Tasks | Availability | Description |
|---------|------|-------|--------------|-------------|
| **CHB-MIT** | 23 subjects | Epilepsy detection | [Link](https://physionet.org/content/chbmit/1.0.0/) | Scalp EEG recordings from pediatric subjects |
| **TUH EEG** | 25,000+ sessions | Multiple | [Link](https://www.isip.piconepress.com/projects/tuh_eeg/) | Largest public EEG corpus |
| **DEAP** | 32 subjects | Emotion recognition | [Link](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) | EEG and peripheral physiological signals |
| **SEED** | 15 subjects | Emotion recognition | [Link](https://bcmi.sjtu.edu.cn/home/seed/) | EEG recordings during emotional film clips |

### fMRI Datasets

| Dataset | Size | Tasks | Availability | Description |
|---------|------|-------|--------------|-------------|
| **ADNI** | 2000+ subjects | Alzheimer's disease | [Link](http://adni.loni.usc.edu/) | Longitudinal neuroimaging data |
| **ABIDE** | 1112 subjects | Autism | [Link](http://fcon_1000.projects.nitrc.org/indi/abide/) | Resting-state fMRI from multiple sites |
| **REST-meta-MDD** | 2428 subjects | Depression | [Link](http://rfmri.org/REST-meta-MDD) | Large-scale depression neuroimaging |
| **HCP** | 1200 subjects | Brain modeling | [Link](https://www.humanconnectome.org/) | High-quality multimodal brain imaging |

## Tools and Resources

### Graph Construction Tools

- **Nilearn**: Machine learning for neuroimaging in Python - [GitHub](https://github.com/nilearn/nilearn)
- **MNE-Python**: MEG and EEG analysis in Python - [GitHub](https://github.com/mne-tools/mne-python)
- **Brain Connectivity Toolbox**: Complex network analysis of brain networks - [Site](https://sites.google.com/site/bctnet/)

### DGNN Frameworks

- **PyTorch Geometric Temporal**: Temporal extension of PyG - [GitHub](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- **DGL (Deep Graph Library)**: Scalable graph neural networks - [GitHub](https://github.com/dmlc/dgl)
- **Spektral**: Graph neural networks with Keras and TensorFlow - [GitHub](https://github.com/danielegrattarola/spektral)

### Visualization Tools

- **BrainNet Viewer**: Brain network visualization - [Site](https://www.nitrc.org/projects/bnv/)
- **Gephi**: Interactive visualization platform for networks - [Site](https://gephi.org/)
- **NetworkX**: Network analysis in Python - [GitHub](https://github.com/networkx/networkx)

## Related Surveys and Resources

- [Graph Neural Networks in Network Neuroscience](https://doi.org/10.1109/TPAMI.2022.3209686) - IEEE TPAMI 2023
- [A Survey of Dynamic Graph Neural Networks](https://doi.org/10.48550/arXiv.2404.18211) - arXiv 2024
- [Awesome Graph Neural Networks](https://github.com/GRAND-Lab/Awesome-Graph-Neural-Networks)
- [Awesome EEG Resources](https://github.com/meagmohit/awesome-eeg)

## Contributing

We welcome contributions! Please feel free to submit a pull request to add new papers, datasets, or tools. When adding papers, please follow the existing format:

```markdown
| Year | [Paper Title](DOI/URL) | Model Type | Data Type | [Code](GitHub URL) or - |
```

### Contribution Guidelines

1. Papers should be directly related to dynamic graph neural networks in neurological applications
2. Include complete citation information (title, authors, venue, year)
3. Provide direct links to paper and code when available
4. Maintain chronological order (newest first) within each section

## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{dgnn_neurological_2025,
  title={Applications of Dynamic Graph Neural Networks in Neurological Function Modeling and Disorder Recognition: A Systematic Review},
  author={Your Authors},
  journal={Your Journal},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact [zjihai360@gmail.com]

---

**Last Updated:** January 2025

**Maintainers:** [Zhang jihai](https://github.com/zhangjihai360)

⭐ If you find this repository helpful, please consider giving it a star!