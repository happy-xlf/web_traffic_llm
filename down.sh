export https_proxy=10.7.4.2:3128
export https_proxy=10.7.4.2:3128
# python save_hf_model.py
# git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b3345
git submodule update --init --recursive
make clean
make all -j
git log -1
cd ..
python save_hf_model.py