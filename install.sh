echo "Installing numpy"
pip3 install numpy

echo "Installing PyTorch"
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

echo "Installing pandas"
pip3 install pandas

echo "Installing tqdm"
pip3 install tqdm