# vit-detect-rotation
detect the orientation of an image with a finetuned vision-tranformer



create virtual environment and install requirements
```
cd vit-detect-rotation
python -m venv .venv
.venv\Scripts\Activate.ps1

```
for cpu:
```
pip install torch torchvision
```
for gpu:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

install requirements
```
pip install -r requirements.txt
```

use from CL
```
python main.py path/to/image_folder --recursive --dry
```  

command line options

| Command | Description | Default |
| ----- | ----- | ----- |
| `--recursive` | If given, search for all **JPG**, **PNG** in the directory and all sub-directories, for any other value search only the directory | Off |
| `--dry` | If given, do nothing, for any other value this has no effect | Off |
| `--cpu` | If given, do nothing, for any other value this has no effect | Off |



Notes:
  - currently CPU only
  - will create a file "log.txt" in csv format inside each subfolder with the following content "num rotated, num untouched,"
