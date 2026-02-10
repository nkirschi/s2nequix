
wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/train_300K.xyz -P data/
wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/train_mixedT.xyz -P data/

wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/test_300K.xyz -P data/
wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/test_600K.xyz -P data/
wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/test_1200K.xyz -P data/
wget https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/refs/heads/main/dataset_3BPA/test_dih.xyz -P data/

python3 data/convert_3bpa.py