# Neural Style Transfer

## Installation

### for CPU usage

```bash
# create conda environment
conda create --name nst_env

# activate environment
conda activate nst_env

# install libraries and packages
pip install -r requirements.txt
```


### for GPU usage

This involves some additional steps to those above. Make sure you have an nvidia GPU available for your setup.

```bash
# create conda environment
conda create --name nst_env

# activate environment
conda activate nst_env

# install cuda tools
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# set up LD_LIBRARY PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# install libraries and packages
pip install -r requirements.txt

# Verify install: this should return message indicating that GPU is available
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Running NST

Make sure you're in the virtual environment described above.

Place your content image in `images/content_images` and style image in `images/style_images` with suitable filenames.

Open `run_nst.py` and alter the default values for `--CONTENT_IMAGE_NAME` and `--STYLE_IMAGE_NAME` those of your content and style image filenames respectively.

Alternatively, run the following on command line:

```bash
python run_nst.py -c <filename of content image> -s <filename of style image>
```

### Note for GPU usage:

If you're running this script on a GPU, you may find that your GPU is out of memory because it has allocated its memory to the previous run. To free your GPU memory up, run the following in command line:

```bash
# run nvidia system managar interface to view running processes
nvidia-smi

# select PID of process you want to kill
sudo kill -9 PID
```

Now you can run the script again.