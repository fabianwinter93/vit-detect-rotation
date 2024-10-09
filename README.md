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
python run.py path/to/image_folder --recursive --dry --quadro --bf16 --verbosity=2
```  

command line options

| Command | Description | Default |
| ----- | ----- | ----- |
| `--recursive` | If given, search for all **JPG**, **PNG** in the directory and all sub-directories, for any other value search only the directory | Off |
| `--VIT80` | Use vit (80 million) | On |
| `--VIT20` | Use vit (20 million) | Off |
| `--ENET20` | Use efficientnet (20 million) | Off |
| `--ENET4` | Use efficientnet (4 million) | Off |
| `--GMIX25` | Use gmixer (25 million) | Off |
|  |  |  |
|  |  |  |
|  |  |  |
| `--quant` | If given, use quantized model (must use **CPU**) (only works with the large model for now) | Off |
| `--onnx` | Use onnx runtime | Off |
| `--onnx-reparam` | reparametrize model | Off |
| `--cpu` | Do not use a gpu (by default gpu is used) | Off |
|  |  |  |
|  |  |  |
| `--dry` | If given, do nothing, for any other value this has no effect | Off |
| `--f32` | Use float32 for everything instead of bf16 (b16 is default) | Off |
| `--verbosity` | Verbosity level. **0** - print all filenames, **1** - print only filenames of rotated images, **2** tqdm progress bar | 2 |
|  |  |  |
|  |  |  |
| `--quadro` | If given, compare logit-confidence score for each rotation and pick the best, else use logits for prediction | Off |
