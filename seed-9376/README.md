# Q_SNet

### Spiking Neural Networks in Quaternion Space: Lightweight Modeling for EEG-Based cross subjects emotion recognition
##### Core idea: Constructing quaternions of EEG signals along the channel dimension + Quaternion rotation as an inductive bias + magnitude-triggered LIF

### News
#### 🎉🎉🎉 We've joined in [braindecode](https://******************.html) toolbox. Use [**here**](https://*****************.html) for detailed info.


Thanks to colleagues for helping with the modifications.

## Abstract
![Network Architecture](/visualization/fig1.png)

## In short, we did three things:

- We used quaternion representation learning in cross-subject EEG signals.
- We found that quaternion rotation can be an effective means of mitigating distribution shift.
- We modified LIF neurons in an attempt to achieve a cross-subject EEG decoding model with fewer parameters and higher computational efficiency. (The results were quite satisfactory.)


## Requirements:
- Python 3.11.4
- Pytorch 2.0.2
- Intel(R) Xeon(R) Gold 6226R CPU and an NVIDIA GeForce RTX 4090 GPU

## Datasets
For evaluation, the leave-one-subject-out cross-validation strategy was adopted.
- [SEED](https://******/iv/) - acc 93.76% (SESSION1)
- [SEED-IV](https://**********/iv/) - acc 78.25% (SESSION3)
- [SEED-V](https://************/seed.html) - acc 89.98% (SESSION3)




