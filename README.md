# robo-learn

To begin with, install **miniconda** and run the first **OR** second set of commands depending on whether you have a GPU

### GPU
```bash
# Create GPU environment (PyTorch + CUDA)
conda env create -f envs/rl-gpu.yml
conda config --set channel_priority strict
conda activate rl-gpu

# Install Python-level dependencies
pip install -r envs/requirements.txt
```

### CPU
```bash
# Create CPU-only environment (PyTorch + cpuonly)
conda env create -f envs/rl-cpu.yml
conda config --set channel_priority strict
conda activate rl-cpu

# Install Python-level dependencies
pip install -r envs/requirements.txt
```
