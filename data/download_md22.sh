
wget https://sgdml.org/secure_proxy.php?file=repo/static/md22_buckyball-catcher.zip -P data/
wget https://sgdml.org/secure_proxy.php?file=repo/static/md22_double-walled_nanotube.zip -P data/

mkdir -p data/md22/buckyball-catcher
mkdir -p data/md22/double-walled_nanotube

unzip data/secure_proxy.php\?file\=repo%2Fstatic%2Fmd22_buckyball-catcher.zip -d data/ -x "__MACOSX/*"
unzip data/secure_proxy.php\?file\=repo%2Fstatic%2Fmd22_double-walled_nanotube.zip -d data/ -x "__MACOSX/*"

python3 data/convert_md22.py

rm data/secure_proxy.php?file=repo%2Fstatic%2Fmd22_buckyball-catcher.zip
rm data/secure_proxy.php?file=repo%2Fstatic%2Fmd22_double-walled_nanotube.zip