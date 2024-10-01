import warnings

warnings.filterwarnings("ignore")

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "./models"

import glob

import argparse

import torch
from PIL import Image 
import timm

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_name = "timm/vit_small_patch16_384.augreg_in21k_ft_in1k"
#ckp = "vit_small_patch16_384.augreg_in21k_ft_in1k_with_orientation_head.pt"
"""
model_name = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
ckp = "vit_base_patch16_224.augreg2_in21k_ft_in1k_with_orientation_head.pt"



model = timm.create_model(
    model_name,
    checkpoint_path=ckp,
    pretrained=True,
    num_classes=4,
)
"""

model = timm.create_model('hf_hub:herrkobold/vit_base_patch16_224.augreg2_in21k_ft_in1k_with_orientation_head.pt', pretrained=True)



data_config = timm.data.resolve_model_data_config(model)
inference_transforms = timm.data.create_transform(**data_config, is_training=False)


def prepare_batch(image, quad):
    images = [image]

    if quad:
        for angle in [90, 180, 270]:
            img = rotate_image_lossless_transpose(image, angle)
            images.append(img)

    images = [inference_transforms(img) for img in images]

    images = torch.stack(images, 0).to(device)
    return images

def predict(batch):
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, -1)
    
    if batch.shape[0] > 1:
        confidence = probs[:, 0]
        pred = torch.argmax(confidence)
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')           
    parser.add_argument('--recursive', default="NO",
                        help="If exactly YES, will find files in subfolders as well. For any other value, will only consider files in the source directory")
    parser.add_argument("--quadro", default="NO", help="If exactly YES, compare logit-confidence score for each rotation and pick the best, else use logits for prediction")
    parser.add_argument("--dry", default="NO", help="If exactly YES, do nothing")
    parser.add_argument("-v", default="2", help="verbosity level")
    
    args = parser.parse_args()

    RECURSIVE = args.recursive == "YES"
    QUADRO = args.quadro == "YES"
    DRY = args.dry == "YES"
    VERBOSE = args.v

    assert VERBOSE in ["0", "1", "2"], "Only levels 0, 1, 2 are valid."

    if QUADRO:
        trace_inp = torch.rand((4, 3, 224, 224))
    else:
        trace_inp = torch.rand((1, 3, 224, 224))
        
    model.eval()    
    model.to(device)
    model = torch.jit.script(model, trace_inp.to(device))
    #model = torch.jit.optimize_for_inference(model)
    model = torch.compile(model, backend="cudagraphs", fullgraph=True)

    src_dir = os.path.abspath(args.source_dir)

    source_files = find_files(src_dir, RECURSIVE)
    #source_files = validate_src_files(source_files)
    n_src_files = len(source_files)

    assert n_src_files > 0, f"No valid files (jpg, png) found in {src_dir}"
    
    print(f"Found {n_src_files} valid files in {src_dir}.")


    predictions = {}

    
    num_images_to_reorient = 0

    with torch.no_grad():

        _range = range(n_src_files)
        if VERBOSE in ["2"]:
            _range = tqdm(_range)
            
        for i in _range:

            fname = source_files[i]
            fpath = os.path.join(src_dir, fname)

            
            img = Image.open(fpath)
            batch = prepare_batch(img, True)
            
            pred = predict(batch)

            
            #logits = model(inference_transforms(img)[None, ...].to(device))
            #probs = torch.nn.functional.softmax(logits, -1)
            #pred = torch.argmax(probs, -1).item()

            predictions[fpath] = pred
    
            if pred == 0:
                if VERBOSE in ["0"]:
                    print(fname)
            else:
                if VERBOSE in ["0", "1"]:
                    print(f"{fname} - turn {90*pred}°")
                    
                num_images_to_reorient += 1
    
                targetpath = fpath
                #targetpath = targetpath.replace(".jpg", "_COPY.jpg")
                #targetpath = targetpath.replace(".png", "_COPY.png")
    
                rotated_img = rotate_image_lossless_transpose(img, 90*pred)
                if DRY is False:
                    rotated_img.save(targetpath)
    
        
    
        print(f"Found {num_images_to_reorient} images with incorrect orientation") 
        if num_images_to_reorient == 0: exit()
    
    
        







