cd ..
git clone https://github.com/565353780/ma-sh.git

cd ma-sh
./compile.sh

pip install -U torch torchvision torchaudio

pip install -U ffmpeg pillow open3d mediapy lpips \
  scikit-image tqdm trimesh plyfile opencv-python \
  tensorboard

cd ../mash-2dgs/submodules/diff-surfel-rasterization
pip install -e .

cd ../simple-knn
pip install -e .
