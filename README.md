# EchoCommand: Audio-Based Speech Classification for Accessibility

This project was created by Peter Wang.

## 1. Introduction

The goal of this project is to develop a machine learning algorithm that takes an audio sample as input and classifies it based on a set of command words. The main motivation behind this project is to enhance the accessibility of services that rely on interactive visual inputs. For instance, many mall directories require users to type inputs on a touchscreen display. This technology would allow visually impaired users to interact with such displays using voice commands.

Machine learning is an effective approach due to the inherent variability in audio samples. Audio from different sources will vary, making it infeasible to explicitly program an algorithm to classify audio. Instead, the best approach is to train a model using a dataset of diverse audio samples.

## 2. Background & Related Work

Audio classification is widely used in fields such as Environmental Sound Classification (ESC) and Audio Generation. Classification can be performed using either raw audio waveforms or 2D spectrograms generated from raw waveforms. One-dimensional convolutional neural networks (1D CNNs), such as the EnvNet architecture, have been used to classify raw audio waveforms. For spectrograms, the classification process is more complex, as spectrograms do not behave like natural images.

Some models employ multi-stream architectures that process both raw audio and spectrograms before aggregating predictions. For ESC, a three-stream network can take raw audio, short-term Fourier transform coefficients, and delta spectrograms as inputs. Other approaches include two-stream networks utilizing mel-spectrograms and mel frequency cepstral coefficients. In some cases, a single-stream network using mel-spectrograms can achieve performance comparable to multi-stream architectures.

## 3. Data Processing

The dataset used in this project is the **Speech Commands Dataset**, created by Google and TensorFlow. It consists of 64,721 audio samples across 30 classes, where each sample corresponds to either a command word (e.g., "stop") or an auxiliary word (e.g., "dog"). Auxiliary words help in detecting unknown commands.

### Dataset Splitting:
- **Training Set:** 48,320 samples (74.7%)
- **Validation Set:** 8,631 samples (13.3%)
- **Testing Set:** 7,770 samples (12.0%)
- **Total:** 64,721 samples

To ensure that validation and testing sets do not contain samples from the same contributors as the training set, a hashing function was used to assign all of a contributor’s samples to a single split.

To handle class imbalances, oversampling was applied. The largest class had 1,791 samples in the training set, which was used as a limit for oversampling. Each class was randomly duplicated until it reached this number.

After splitting, each sample was:
1. Converted into a tensor
2. Augmented using audio data augmentation techniques
3. Padded or trimmed to a uniform duration of one second
4. Converted into a spectrogram
5. Normalized to match the smallest spectrogram dimensions in the dataset

Data augmentation was performed using the Sox effect library in TorchAudio. Background noise samples were normalized in loudness, trimmed to one second, and overlaid onto audio samples for training.

## 4. Model Architecture

A convolutional neural network (CNN) was used for classification, taking in spectrograms of size **[3, 201, 30]** as input. CNNs are known to perform well for large-scale audio classification tasks.

### Model Design:
- Two convolution layers
- Max pooling layers
- Dropout layers to reduce overfitting
- Kernel size of 5
- ReLU activation function for fully connected layers

## 5. Baseline Model

A Random Forest classifier was used as a baseline. It took in one-second audio samples, converted into numpy arrays. Samples shorter than one second were removed, reducing the dataset to **58,252 samples**.

Using **StandardScaler()** from Scikit-Learn, audio samples were transformed before training the **RandomForestClassifier** with 20 trees. More trees yielded minimal performance improvements.

Results:
- **Training Accuracy:** 99.4%
- **Testing Accuracy:** 3.3%

This poor testing accuracy suggests significant overfitting.

## 6. Quantitative Results

The CNN model achieved an overall accuracy of **76.3%** on the testing set. The validation and training accuracies were above **75%**. 

Observations:
- Accuracy curves flattened around **45 epochs**, suggesting training saturation.
- Increased divergence between training and validation accuracy after **20 epochs** indicates overfitting.
- Words with similar pronunciations (e.g., "No" and "Go") had lower precision due to spectrogram similarities.

## 7. Qualitative Results

Saliency maps were analyzed to understand the model’s decision-making process. These maps highlight the most influential parts of the spectrogram for classification. 

Findings:
- The model relies on darker regions of the spectrograms.
- Incorrect predictions occur when input spectrograms resemble those of another class.
- Commands such as "No" and "Go" showed lower precision due to phonetic similarity.

## 8. Evaluating on New Data

To test the model on real-world data, 4 samples per class were recorded, resulting in **480 total samples**. The recorded audio was processed similarly to dataset samples.

Results:
- Accuracy on new data: **75.4%**
- Performance was consistent with test set accuracy, suggesting generalizability.

## 9. Ethical Considerations

The dataset does not provide demographic information about contributors. As a result:
- The model may be biased towards certain accents or voice characteristics.
- Gender-based bias may exist due to natural differences in male and female voices.
- Addressing these biases would require demographic labels, which are unavailable due to privacy concerns.

## 10. Discussion

### Strengths:
- **Significant improvement** over the baseline model
- **Consistent accuracy** across different datasets
- **Useful insights** gained from saliency maps and qualitative analysis

### Weaknesses:
- Struggles with **similar-sounding words** (e.g., "No" and "Go")
- Overfitting observed beyond **20 epochs**
- Possible **accent and gender biases** in classification

To improve user experience, an additional feature could be implemented to confirm predictions before taking action. However, this could be inconvenient for users, especially those who are hard of hearing.

## 11. Future Improvements

### Transfer Learning:
- Attempted using **ResNet18**, but results were poor due to incompatibility with spectrograms.
- Object detection models may not be suitable for analyzing black-and-white spectrograms.

### Data Augmentation:
- Helped reduce overfitting.
- Future models could explore **dynamic time warping** or **phoneme-level augmentation** to improve accuracy on similar-sounding words.

## 12. References

[1] Y. Tokozume and T. Harada, "Learning environmental sounds with end-to-end convolutional neural network," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 2017, pp. 2721-2725, doi: 10.1109/ICASSP.2017.7952651.

[2] Palanisamyy K., Singhaniay D., Yaoy A., “Rethinking CNN Models for Audio Classification”, arXiv:2007.11154v2, 2020.

[3] Li X., Chebiyyam V., Kirchhoff K., "Multi-stream Network With Temporal Attention For Environmental Sound Classification", arXiv:1901.08608, 2020.

[4] Su Y., Zhang K., Wang J., Madani K., "Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion".

[5] Hershey S., “CNN Architectures for Large-Scale Audio Classification”.

