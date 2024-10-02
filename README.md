# vit-detect-rotation
detect the orientation of an image with a finetuned vision-tranformer



create virtual environment and install requirements
```
cd vit-detect-rotation
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

use from CL

```
python run.py path/to/image_folder --recursive --dry --quadro --bf16 -v=2
```  

command line options

| Command | Description |
| --- | --- |
| `--recursive` | If given, search for all **JPG**, **PNG** in the directory and all sub-directories, for any other value search only the directory |
| `--quant` | If given, use quantized model (must use only) |
| `--dry` | If given, do nothing, for any other value this has no effect |
| `--cpu` | Do not use a gpu (by default gpu is used) |
| `--f32` | Use float32 for everything instead of bf16 (b16 is default) |
| `--verbosity` | Verbosity level. **0** - print all filenames, **1** - print only filenames of rotated images, **2** tqdm progress bar |

| `--quadro` | If given, compare logit-confidence score for each rotation and pick the best, else use logits for prediction |
| `--compile` | Use torch.compile on the model |
| `--script` | Use torch.jit.script on the model |
| `--trace` | Use torch.jit.trace on the model |
