# Visual Transformer (ViT) from scratch: Implementing  "An Image is Worth 16x16 Words"

This repository delves deep into the transformer architecture, focusing on its use in computer vision
The implementation is based on the
paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) and has been crafted entirely from scratch for
educational purposes. Additionally, the performance of the Visual Transformer has been compared with a ResNet18 model to
provide a comprehensive evaluation of its capabilities.

## 🌐 Introduction to Transformer Architecture

Introduced by Vaswani et al. in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), the
transformer architecture revolutionized the field of natural language processing with its self-attention mechanism.
Unlike traditional RNNs or CNNs, transformers operate on the entire sequence of data simultaneously, capturing long-term
dependencies and intricate patterns in data.
Thanks to its design, the architecture can run in parallel, making it ideal for modern hardware accelerators.

## 📸 Transformers in Computer Vision

Though transformers were initially designed for sequence-to-sequence tasks such as machine translation, their
versatility led to adaptations for computer vision tasks. The Visual Transformer (ViT) processes images using the
transformer architecture by dividing them into fixed-size non-overlapping patches, linearly embedding them, and feeding
them as sequences into the transformer. This model captures both local and global patterns in an image, challenging
traditional convolution-based architectures in performance.

Here's a brief breakdown of how ViT works:

1. **Image Patching**: Input images are divided into fixed-size non-overlapping patches, akin to "words" or "tokens" in
   NLP.
2. **Linear Embedding**: Each patch undergoes linear embedding to achieve a flattened representation with a specified
   dimension.
3. **Positional Encodings**: To maintain positional data, positional encodings are added to these embeddings, mirroring
   NLP transformer techniques.
4. **Transformer Encoders**: The embeddings are processed through several transformer encoder layers, capturing both
   local and global patch interactions.
5. **Classification Head**: The transformer output is then directed to a classification head, producing outputs such as
   class probabilities for image classification tasks.

Below is the original image depicting the ViT architecture:

![ViT Architecture](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

## 🛠️ Training and Hyperparameters

### Training the Models

The train.py script makes it easy to train the ViT model and takes command-line arguments to set model configurations.
By default, the script trains the ViT model using the hyperparameters specified
in `configs/vit_custom.yaml`.

To train the models using the provided scripts, follow the steps below:

1. Navigate to the repository's root directory.
2. Execute the training script. Ensure you specify the desired model (either 'vit' or 'resnet'):

   ```
   python train.py --config_path [CONFIG_PATH] --model_type [MODEL_TYPE]
   ```

Parameters:

- `--config_path`: Specifies the path to the YAML configuration file. The default is `configs/vit_custom.yaml`.
- `--model_type`: Indicates the type of model to train. By default, it's set to `vit_custom`.

### Model Configuration

Both models come with dedicated configuration files to easily modify the training and model parameters. These YAML files
allow for a straightforward way to adjust hyperparameters without diving into the code. To customize the model or
training configurations open the respective configuration file (`configs/vit.yaml` for ViT and `configs/resnet.yaml` for
ResNet).

#### Visual Transformer Configuration - `configs/vit.yaml`

**model_config**:

- `patch_size`: Size of the patches the input image is divided into.
- `latent_dim`: Dimensionality of the latent space.
- `n_layers`: Number of transformer layers.
- `n_heads`: Number of attention heads in the multi-head self-attention mechanism.
- `dropout`: Dropout rate for regularization.

#### ResNet Configuration - `configs/resnet.yaml`

**model_config**:

- `resnet_size`: The size of the ResNet architecture (e.g., 18 for ResNet18).
- `pretrained`: Flag to determine if the model should use pretrained weights (true/false).

## 📊 Performance Comparison

### Overview

The performance of the from-scratch Visual Transformer is benchmarked against the classic ResNet18 architecture (not
pretrained) on the Fashion MNIST dataset. The table below provides a comparative overview:

| Model              | Accuracy   | # Parameters (M) |
|--------------------|------------|------------------|
| Visual Transformer | 88.17%     | 7.7              |
| ResNet18           | **88.23**% | 11.2             |

ViT Hyperparameters:

- Patch Size: 7
- Latent Dimension: 320
- Number of Layers: 12
- Number of Heads: 10
- Dropout: 0.3

### Detailed Training Logs

For an in-depth look at the training history, check out the logs and visuals
on [Weights & Biases Report](https://wandb.ai/gianlucagiudice/ViT-Scratch/reports/ViT-from-scratch--Vmlldzo1NjczODAy).

### Model Training and Comparison

The `train_comparison.py` script is designed for a side-by-side comparison of the Visual Transformer and ResNet18
models. To utilize it, provide paths to the appropriate YAML configuration files via command-line arguments:

- `--vit_config_path`: Path for the Visual Transformer's configuration.
- `--resnet_config_path`: Path for the ResNet18's configuration.

Execute the following command to initiate the training and testing process for both models:

```
python train_comparison.py --vit_config_path configs/vit.yaml --resnet_config_path configs/resnet.yaml
```

During this process, both models will be trained and subsequently evaluated on the Fashion MNIST dataset. Comprehensive
training details and test performance metrics will be logged on Weights & Biases.

## ✍ Conclusion

The main goal of this project was to dive deep into the transformer architecture for computer vision and see how it
stacks up against typical CNNs like ResNet.

### ViT in Broader Context

In more extensive studies and on larger datasets, Visual Transformers (ViT) have often exhibited comparable or even
superior performance to state-of-the-art CNNs. Their capacity to capture long-range dependencies in images, without
being constrained by the local receptive fields of CNNs, allows them to excel especially when supplied with vast amounts
of data. The transformer's inherent capability to focus on various parts of an image, depending on the context, makes it
a powerful architecture for computer vision tasks on a grand scale.

### Analysis on Fashion MNIST Results

However, in this specific experiment with the Fashion MNIST dataset, ResNet18 had a slight edge over the Visual
Transformer. Several
factors might contribute to this observation:

1. **Dataset Complexity**: Fashion MNIST, while being more intricate than the original MNIST, remains relatively simple.
   ResNet architectures excel at capturing hierarchical patterns in images, making them potentially more suitable for
   such datasets.

2. **ViT's Need for Data**: ViT's design might demand larger datasets to optimize its extensive parameters fully. On
   datasets like Fashion MNIST, it might not manifest its full potential.

3. **Positional Encodings**: The efficacy of ViT's positional encodings, especially on smaller datasets, could be a
   determinant.

4. **Hyperparameter Sensitivity**: Transformers can be more sensitive to hyperparameters. The settings might have been
   more optimized for ResNet18 in this experiment.

In essence, while ViT shows promising capabilities, it's essential to understand its strengths and weaknesses concerning
the problem at hand. Traditional CNNs like ResNet might still be more effective for smaller datasets, emphasizing the
importance of choosing the right model for the specific task and dataset.

**It's all about picking the right tool for the job. Even if attention is all you need, it might not be all you want.**
