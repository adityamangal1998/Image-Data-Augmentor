echo -e "----------------Installing Image Data Augmentor----------------\n\n"
python -m pip install --upgrade build
echo -e "----------------Removing Distribution Folder----------------\n\n"
rm -rf dist
echo -e "----------------Building Distribution Folder----------------\n\n"
python -m build
LATEST_VERSION=$(find dist -type f -name "*.gz")
echo -e "----------------Installing----------------\n\n"
pip install $LATEST_VERSION
echo -e "----------------Installing Required Libraries----------------\n\n"
pip install -r requirements.txt
echo -e "----------------Installation Done----------------\n\n"
