#  Cross-modal Feature Disentangling via Bidirectional Distillation for Multimodal Recommendation


<p align="left">
    <img src='https://img.shields.io/badge/key word-Recommender Systems-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Collaborative Filtering-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Multimodal User Preference-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Bidirectional Distillation-green.svg' alt="Build Status">
</p>

Multimodal recommendation (MMRec) models typically integrate the multimodal content of items (e.g., images, text) with collaborative signals (i.e., behavioral similarity reflected by interaction data) to capture user preferences. However, the prevailing models predominantly encapsulate intricate multimodal content within an integrated, albeit implicit, embedding space. This approach hinges on a singular set of latent representations to assimilate and process information across diverse modalities, despite the fact that these modalities each possess distinct characteristics as well as shared similarities in both their features and underlying semantics. We argue that this simplified embedding learning approach fails to provide fine-grained representations of the diverse and complex multimodal features, leading to the entanglement and confounding of diverse preference cues. As a matter of fact, different modalities describe the unique aspects and diversity of items from various perspectives (e.g., images reflect overall silhouette, while text conveys usage), yet they also convey commonalities and overlaps through similar aspects (e.g., both can depict style and color). Disentangling these multi-aspect (distinct aspect and overlapping aspect) features and learning finer-grained representations is crucial for complete capture and coherent expression of user preferences. To address this, we develop a novel model, Cross-modal Feature DisenTangling via Bidirectional Distillation (CFDTBD), which explicitly disentangles multimodal user preference modeling into (1) differential feature learning and (2) common feature learning across different modalities. The former captures unique, complementary features from different modalities, providing diverse preference expressions, while the latter identifies consistent, overlapping features, reinforcing shared user interests. By disentangling these two types of features, CFDTBD effectively extracts cross-modal differential and common preference cues from otherwise entangled and coarse item embeddings, enabling a fine-grained characterization of user preferences, based on the relative contributions of distinct and overlapping aspects in multimodal content. Therein, a bidirectional distillation strategy is formulated to ensure multi-aspect feature learning by pushing the embedding space of differential features farther apart while bringing common features closer together. Extensive experiments on three public datasets validate the effectiveness of CFDTBD.

### Before running the codes, please download the [datasets](https://www.aliyundrive.com/s/V1RPArCZQYt) and copy them to the Data directory.

## Prerequisites

- Tensorflow 1.10.0
- Python 3.6
- NVIDIA GPU + CUDA + CuDNN

