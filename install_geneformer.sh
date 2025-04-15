pip install "ray[default]@https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp313-cp313-manylinux2014_x86_64.whl" # only for python 3.13
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
pip install transformers==4.49.0
