#!/bin/sh
# Configuring the software
sudo apt update -y
sudo apt install -y cmake g++ libncurses5-dev libgmp-dev libflint-dev libflint-arb-dev nvidia-cuda-toolkit mpich libmpich-dev
sudo apt-get install -y gcc make linux-headers-$(uname -r)

# Downloading the CUDA driver
if ec2metadata | grep -q p2 || ec2metadata | grep -q p3 ; then
    wget http://us.download.nvidia.com/tesla/418.40.04/NVIDIA-Linux-x86_64-418.40.04.run
elif ec2metadata | grep -q g3 ; then
    aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
elif ec2metadata | grep -q g2 ; then
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/418.56/NVIDIA-Linux-x86_64-418.56.run
else
    echo "Instance type is neither p2, p3, g2, g3. Aborting"
fi

# Installing the cuda driver
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run

# Setting up configuration script for GPU
echo 'if [ -f /tmp/gpu_activated ] ; then
    echo "GPU already configured"
else
    echo "Configuring GPU for first time"
    touch /tmp/gpu_activated
    sudo nvidia-persistenced
    sudo nvidia-smi --auto-boost-default=0' >> ~/.profile

if ec2metadata | grep -q p2 ; then
    echo '    sudo nvidia-smi -ac 2505,875' >> ~/.profile
elif ec2metadata | grep -q p3 ; then
    echo '    sudo nvidia-smi -ac 877,1530' >> ~/.profile
elif ec2metadata | grep -q g3 ; then
    echo '    sudo nvidia-smi -ac 2505,1177' >> ~/.profile
else
    echo "Instance is g2"
fi

echo 'fi' >> ~/.profile
echo "Rebooting now..."
sleep 1
sudo reboot
