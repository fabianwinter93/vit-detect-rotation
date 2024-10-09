
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

from concurrent import futures

MODEL_NAME_80M = 'hf_hub:herrkobold/vit_base_patch16_224.augreg2_in21k_ft_in1k_with_orientation_head.pt'
MODEL_NAME_4M = 'hf_hub:herrkobold/efficientnet_b0.ra_in1k_with_orientation_head'
MODEL_NAME_20M = "hf_hub:herrkobold/vit_small_patch16_224.dino.with_orientation_head"
MODEL_NAME_17M = 'hf_hub:herrkobold/efficientnet_b4.ra2_in1k_2_with_orientation_head'

MODEL_SIZE_URL = {"L":MODEL_NAME_80M, "M": MODEL_NAME_20M, "S": MODEL_NAME_17M, "XS": MODEL_NAME_4M}


MODEL = None
TRANSFORM = None

ONNX_SESSION = None
ONNX_INPUT_NAME = None


def load_model(model_name):
    global MODEL, TRANSFORM
    model = timm.create_model(model_name, pretrained=True, num_classes=4)
    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    inference_transforms = timm.data.create_transform(**data_config, is_training=False)
    MODEL = model
    TRANSFORM = inference_transforms
    


def load_quant_model(model_name):
    load_model(model_name)

    torch.quantization.quantize_dynamic(
        MODEL, 
        {torch.nn.Linear},  # Specify layers to quantize
        dtype=torch.qint8,    # Use int8 quantization
        inplace=True,
    )
    
    
    
def load_onnx_model(model_name):
    global MODEL, ONNX_SESSION, ONNX_INPUT_NAME

    import onnxruntime as ort
    from timm.utils.onnx import onnx_export
    from timm.utils.model import reparameterize_model

    
    load_model(model_name)

    MODEL = reparameterize_model(MODEL)
    onnx_file_name = model_name.removeprefix("hf_hub:herrkobold/") + ".onnx"
    
    if os.path.exists(f"./models/{onnx_file_name}"):
        pass
    else:
        onnx_export(MODEL.eval(), f"./models/{onnx_file_name}",
            torch.rand((1, 3, 224, 224), requires_grad=False), 
            training=False, 
            check=True, 
            check_forward=False)
        
        ONNX_SESSION = ort.InferenceSession(f"./models/{onnx_file_name}")
        ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    

def prepare_batch(image, quad):
    images = [image]

    if quad:
        for angle in [90, 180, 270]:
            img = rotate_image_lossless_transpose(image, angle)
            images.append(img)

    images = [TRANSFORM(img.convert("RGB")) for img in images]

    images = torch.stack(images, 0).to(device)
    return images

def predict(batch):
    if ONNX_SESSION is not None:
        logits = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: batch})
    else:
        logits = MODEL(batch)
    
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
    parser.add_argument("--XS", action='store_true')
    parser.add_argument("--S", action='store_true')
    parser.add_argument("--M", action='store_true')
    parser.add_argument("--L", action='store_true')

        
    parser.add_argument('--recursive', action="store_true", help="If exactly YES, will find files in subfolders as well. For any other value, will only consider files in the source directory")
    parser.add_argument("--quadro", action='store_true', help="If exactly YES, compare logit-confidence score for each rotation and pick the best, else use logits for prediction")
    parser.add_argument("--dry", action='store_true', help="If exactly YES, do nothing")
    
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--onnx", action='store_true')
    
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--trace", action='store_true')
    parser.add_argument("--script", action='store_true')

    parser.add_argument("--f32", action='store_true')
    
    parser.add_argument("--verbosity", default="2", help="verbosity level")
    
    args = parser.parse_args()

    CPU = args.cpu or args.quant

    if args.L: model_name = MODEL_SIZE_URL["L"]
    elif args.M: model_name = MODEL_SIZE_URL["M"]
    elif args.S: model_name = MODEL_SIZE_URL["S"]
    elif args.XS: model_name = MODEL_SIZE_URLS["XS"]
    else:
        raise Exception("Must choose a model size [S, M, L]")
    
    if CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    
    if args.quant:
        load_quant_model(model_name)
    elif args.onnx:
        load_onnx_model(model_name)
    else:
        load_model(model_name)


    #num_params = sum(p.numel() for p in model.parameters()) / (10**6)
    #print(f"load model: {model_name} with {num_params} million parameters")

    RECURSIVE = args.recursive
    QUADRO = args.quadro
    DRY = args.dry
    VERBOSE = args.verbosity
    

    assert VERBOSE in ["0", "1", "2"], "Only levels 0, 1, 2 are valid."

    
    if args.f32 or args.quant:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
        MODEL.bfloat16()
        
    MODEL.to(device)


    

    
    with torch.inference_mode():
        
        if args.compile:
            MODEL = torch.compile(MODEL, backend="cudagraphs", fullgraph=True)
    
        if args.script:
            if QUADRO:
                trace_inp = torch.rand((4, 3, 224, 224))
            else:
                trace_inp = torch.rand((1, 3, 224, 224))
            MODEL = torch.jit.script(MODEL, trace_inp.to(dtype).to(device))
        elif args.trace:
            if QUADRO:
                trace_inp = torch.rand((4, 3, 224, 224))
            else:
                trace_inp = torch.rand((1, 3, 224, 224))
            MODEL = torch.jit.trace(MODEL, trace_inp.to(dtype).to(device))
    
    
    
        src_dir = os.path.abspath(args.source_dir)
        src_dir = os.path.normpath(src_dir)
    
        source_files = find_files(src_dir, RECURSIVE)
        #source_files = validate_src_files(source_files)
        n_src_files = len(source_files)
    
        assert n_src_files > 0, f"No valid files (jpg, png) found in {src_dir}"
        
        print(f"Found {n_src_files} valid files in {src_dir}.")
    
    
    
        subfolders = {}
        for fname in source_files:
            dirname = os.path.dirname(fname)
            if dirname not in subfolders:
                subfolders[dirname] = []
    
            subfolders[dirname].append(fname)
    
    
        for dirname, files in subfolders.items():
            num_rotated = 0
            num_untouched = 0
            
    
            for fname in tqdm(files, total=len(files), desc=dirname):
                
                fpath = os.path.join(src_dir, fname)
        
                
                img = Image.open(fpath)
                batch = prepare_batch(img, True).to(dtype)
                
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
                    #targetpath = targetpath.replace(".jpg", "_COPY.jpg")
                    #targetpath = targetpath.replace(".png", "_COPY.png")
        
                    rotated_img = rotate_image_lossless_transpose(img, 90*pred)
                    if DRY is False:
                        rotated_img.save(targetpath)
        
    
            if DRY is False:
                with open(os.path.join(dirname, "log.txt"), "w") as wf:
                        
                    wf.write(f"rotated,untouched,\n")
                    wf.write(f"{num_rotated},{num_untouched},\n")
                
            print(f"Found {num_rotated}/{num_untouched+num_rotated} images with incorrect orientation") 
            #if num_images_to_reorient == 0: exit()
        

        
