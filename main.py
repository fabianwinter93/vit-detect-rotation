import warnings
warnings.filterwarnings("ignore")

import os

try:
    os.mkdir("./models")
except FileExistsError:
    pass

os.environ["HUGGINGFACE_HUB_CACHE"] = "./models"

import glob
import argparse
from PIL import Image 
from tqdm import tqdm

import torch
torch.set_grad_enabled(False)

import timm

VIT80 = 'hf_hub:herrkobold/vit_base_patch16_224.augreg2_in21k_ft_in1k_with_orientation_head.pt'
#ENET4 = 'hf_hub:herrkobold/efficientnet_b0.ra_in1k_with_orientation_head'
#VIT20 = "hf_hub:herrkobold/vit_small_patch16_224.dino.with_orientation_head"
#ENET20 = 'hf_hub:herrkobold/efficientnet_b4.ra2_in1k_2_with_orientation_head'
#GMIX25 = 'hf_hub:herrkobold/gmixer_24_224.ra3_in1k.detect_rotation'

MODEL_NAME = VIT80

MODEL = None
TRANSFORM = None


ONNX_SESSION = None
ONNX_REPARAM = True


ONNX = False

DTYPE = torch.float32
DEVICE = "cpu"



def load_model(model_name):
    global MODEL, TRANSFORM

    model = timm.create_model(model_name, pretrained=True, exportable=True, num_classes=4)
    model.eval()
    model.to(DEVICE)
    
    data_config = timm.data.resolve_model_data_config(model)
    inference_transforms = timm.data.create_transform(**data_config, is_training=False)
    MODEL = model
    TRANSFORM = inference_transforms
    
    
    
def load_onnx_model(model_name):
    global MODEL, ONNX_SESSION, ONNX_REPARAM

    import onnxruntime as ort
    from timm.utils.onnx import onnx_export
    

    load_model(model_name)

    from timm.utils.model import reparameterize_model
    MODEL = reparameterize_model(MODEL)
    onnx_file_name = model_name.removeprefix("hf_hub:herrkobold/") + ".reparam.onnx"
    
    if os.path.exists(f"./models/{onnx_file_name}"):
        pass
    else:
        input_size = MODEL.pretrained_cfg["input_size"][-1]

        onnx_export(MODEL.eval(), f"./models/{onnx_file_name}",
            torch.rand((1, 3, input_size, input_size), requires_grad=False), 
            training=False, 
            check=True, 
            check_forward=False,
            input_names=["input0"],
            use_dynamo=False,
            verbose=False)
        
    ONNX_SESSION = ort.InferenceSession(f"./models/{onnx_file_name}")
    del MODEL
    

def prepare_batch(image):
    images = [image]

    images = [TRANSFORM(img.convert("RGB")) for img in images]

    images = torch.stack(images, 0).to(DEVICE)
    return images

def predict(batch):
    
    logits = ONNX_SESSION.run(None, {"input0": batch.to(torch.float32).to(DEVICE).numpy()})
    logits = torch.Tensor(logits)

    probs = torch.nn.functional.softmax(logits, -1)
    
    
    pred = torch.argmax(probs, -1)
    
    return pred.squeeze().item()

def validate_src_files(filenames):
    return [fn for fn in filenames if fn.endswith("jpg") or fn.endswith("png")]

def rotate_image_lossless_transpose(image, angle):
    if angle == 90:
        rotated_img = img.transpose(Image.ROTATE_270)  # ROTATE_270 means rotating 90 degrees clockwise
    elif angle == 180:
        rotated_img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        rotated_img = img.transpose(Image.ROTATE_90)  # ROTATE_90 means rotating 270 degrees clockwise
    
    return rotated_img


def find_files(dirpath, recursive):
    if recursive:
        return glob.glob(f'{dirpath}/**/*.png', recursive=True) + glob.glob(f'{dirpath}/**/*.jpg', recursive=True)
    else:
        return glob.glob(f'{dirpath}/*.png') + glob.glob(f'{dirpath}/*.jpg')




def start_processing_folder_with_model(dirname, recursive, dry, num_files_callback=None, per_image_callback=None, per_folder_callback=None):
    with torch.inference_mode():
        
        src_dir = os.path.abspath(dirname)
        src_dir = os.path.normpath(src_dir)
    
        source_files = find_files(src_dir, recursive)

        n_src_files = len(source_files)
        if num_files_callback is not None:
            num_files_callback(n_src_files)

        assert n_src_files > 0, f"No valid files (jpg, png) found in {src_dir}"
        
        print(f"Found {n_src_files} valid files in {src_dir}.")
    
    
        subfolders = {}
        for fname in source_files:
            dirname = os.path.dirname(fname)
            if dirname not in subfolders:
                subfolders[dirname] = []
    
            subfolders[dirname].append(fname)
    

        file_count_total = 0
        file_count_folder = 0
        folder_count = 0

        for dirname, files in subfolders.items():
            file_count_folder = 0

            folder_count += 1

            num_rotated = 0
            num_untouched = 0
            
            if __name__ == "__main__":
                _range = tqdm(files, total=len(files), desc=dirname)
            else:
                _range = files

            for fname in _range:
                file_count_folder += 1
                file_count_total += 1
                
                fpath = os.path.join(src_dir, fname)
                
                img = Image.open(fpath)
                batch = prepare_batch(img).to(DTYPE)
                
                pred = predict(batch)
        
        
                if pred == 0:
                    num_untouched += 1
                    if VERBOSE in ["0"]:
                        print(fname)
                else:
                    num_rotated += 1
                    if VERBOSE in ["0", "1"]:
                        print(f"{fname} - turn {90*pred}°")
                        
        
                    targetpath = fpath
    
                    rotated_img = rotate_image_lossless_transpose(img, 90*pred)
                    if dry is False:
                        rotated_img.save(targetpath)
        
            if dry is False:
                with open(os.path.join(dirname, "log.txt"), "w") as wf:
                        
                    wf.write(f"rotated,untouched,\n")
                    wf.write(f"{num_rotated},{num_untouched},\n")
                
            print(f"Found {num_rotated}/{num_untouched+num_rotated} images with incorrect orientation") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')           

    parser.add_argument('--recursive', action="store_true")
    
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--verbosity", default="2")
    parser.add_argument("--dry", action='store_true')

    args = parser.parse_args()

    CPU = True #args.cpu or not torch.cuda.is_available()
    
    RECURSIVE = args.recursive
    DRY = args.dry
    VERBOSE = args.verbosity
    
    if CPU:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    print(f"Using {DEVICE}")
    
    
    load_onnx_model(MODEL_NAME)
            

    
    
    start_processing_folder_with_model(args.source_dir, RECURSIVE, DRY)

        

        
