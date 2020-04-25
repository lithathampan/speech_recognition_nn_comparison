# speech_recognition_nn_comparison
Comparisons between Neural Networks with LibriSpeech Speech Recognition

Currently a Work in Progress..

# GPU Setup

Remove CUDA if it is not 10.1
```bash
sudo apt-get --purge remove "*cublas*" "cuda*"

sudo apt-get --purge remove "nvidia-driver-*"
```

Install CUDA 10.1 to be compatible with TF 2.1

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-430
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.4.38-1+cuda10.1  \
    libcudnn7-dev=7.6.4.38-1+cuda10.1
```
# Audio I/O Library Installation

```bash
sudo apt-get install -y portaudio19-dev
```

# Setup

Create a new virtual environment 

```bash
mkvirtualenv SRNC -p python3.6`
```

Install required packages

```bash
pip install -r requirements.txt
```
Remove existing in case of reinstalls

```bash
 pip freeze | xargs pip uninstall -y
 ```



# Data Set up


Download LibriSpeech


