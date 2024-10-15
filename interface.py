import os

from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

import time

import threading

from main import (
    MODEL_NAME, load_onnx_model, find_files, Image, prepare_batch, predict, DTYPE
    )

load_onnx_model(MODEL_NAME)

## globals ##

PROCESS = None
RUNNING = False


ROOT = Tk()
ROOT.resizable(0,0)
CONTENT = Frame(ROOT)



#PROCESSING_WINDOW = None
#PROCESSING_WINDOW_CONTENT = None


#def processing_window_close_fn():
#    global RUNNING, PROCESSING_WINDOW, PROCESSING_WINDOW_CONTENT
#    RUNNING = False
#    PROCESSING_WINDOW.destroy()
#    PROCESSING_WINDOW = None
#    PROCESSING_WINDOW_CONTENT = None
#
#    btn_process_or_pause["state"] = "enabled"





#def openwindow():    
#    global CONTENT, PROCESSING_WINDOW, PROCESSING_WINDOW_CONTENT
#    PROCESSING_WINDOW = Toplevel(CONTENT)
#    PROCESSING_WINDOW.resizable(0,0)
#
#    PROCESSING_WINDOW.protocol("WM_DELETE_WINDOW", processing_window_close_fn)
#
#    PROCESSING_WINDOW_CONTENT = Frame(PROCESSING_WINDOW)
#
#    pbar_progress = Progressbar(PROCESSING_WINDOW, orient=HORIZONTAL, length=300, mode='determinate')
#    btn_pause = Button(PROCESSING_WINDOW, text="pause")
#    btn_stop = Button(PROCESSING_WINDOW, text="stop")
#
#    PROCESSING_WINDOW_CONTENT.grid(column=0, row=0)
#    pbar_progress.grid(column=1, row=0)
#    btn_pause.grid(column=0, row=0)
#    btn_stop.grid(column=2, row=0)


class Process:
    def __init__(self, dirname, recursive, dry):
        
        self.dry = dry
        self.recursive = recursive

        src_dir = os.path.abspath(dirname)
        self.src_dir = os.path.normpath(src_dir)

        self.files = find_files(self.src_dir, self.recursive)
        self.n_files = len(self.files)

        self.PAUSE = False
        self.STOP = False


    def stop(self):
        self.STOP = True
    
    def pause(self):
        self.PAUSE = not self.PAUSE
        
    def rotate(self, image, rotation):
            
        if rotation == 90:
            rotated_image = image.transpose(Image.ROTATE_270)  # ROTATE_270 means rotating 90 degrees clockwise
        elif rotation == 180:
            rotated_image = image.transpose(Image.ROTATE_180)
        elif rotation == 270:
            rotated_image = image.transpose(Image.ROTATE_90)  # ROTATE_90 means rotating 270 degrees clockwise
        
        return rotated_image

    def run(self):
        
        pbar_progress["maximum"] = self.n_files

        subfolders = {}
        for fname in self.files:
            dirname = os.path.dirname(fname)
            if dirname not in subfolders:
                subfolders[dirname] = []

            subfolders[dirname].append(fname)

        count = 0    
        n_rotated = 0
        progress_var.set(count)




        for dirname, files in subfolders.items():


            if os.path.exists(os.path.join(dirname, "log.txt")) and bvar_force.get() is False:
                count += len(files)
                progress_var.set(count)
                continue


            local_total = len(files)
            local_count = 0
            local_rotated = 0

            
            STATUS = None

            for fname in files:
                
                # Check for STOP condition
                if self.STOP:
                    count_var.set(f"processed {count}/{len(self.files)}, rotated {n_rotated} STOPPED")
                    STATUS = "stopped"
                    break
                
                # Pause the loop if PAUSED is True
                while self.PAUSE:
                    time.sleep(0.1)  # Small sleep to prevent CPU overuse
                    

                
                fpath = os.path.join(self.src_dir, fname)
                img = Image.open(fpath)
                batch = prepare_batch(img).to(DTYPE)
                
                pred = predict(batch)



                if pred != 0:
                    n_rotated += 1
                    local_rotated += 1


                if self.dry or pred == 0:
                    pass
                else:
                    rotated_image = self.rotate(img, pred*90)
                    rotated_image.save(fname)


                count += 1
                local_count += 1
                progress_var.set(count)
                

                count_var.set(f"processed {count}/{len(self.files)}, rotated {n_rotated}")
                ROOT.update_idletasks()
            
            if self.STOP:
                count_var.set(f"processed {count}/{len(self.files)}, rotated {n_rotated} STOPPED")
                STATUS = "stopped"
                break

            STATUS = "finished"

            with open(os.path.join(dirname, "log.txt"), "w") as wf:
                wf.write("total,processed,rotated,status,\n")
                wf.write(f"{local_count},{local_total},{local_rotated},{STATUS},\n")

        
        self.FINISHED = not self.STOP
        if self.FINISHED:
            count_var.set(f"processed {count}/{len(self.files)}, rotated {n_rotated} FINISHED")

        end_processing()



def start_processing():
    global RUNNING, PROCESS
    btn_process_or_pause["text"] = "pause"
    RUNNING = True
    btn_stop["state"] = "enabled"
    btn_select_folder["state"] = "disabled"

    ckbtn_recursive["state"] = "disabled"
    ckbtn_dry["state"] = "disabled"
    ckbtn_force["state"] = "disabled"

    count_var.set("")

    PROCESS = Process(folder_path.get(), bvar_recursive.get(), bvar_dry.get())

    threading.Thread(target=PROCESS.run).start()


def end_processing():
    global RUNNING, PROCESS
    btn_process_or_pause["text"] = "process"
    btn_stop["state"] = "disabled"
    btn_select_folder["state"] = "enabled"

    ckbtn_recursive["state"] = "enabled"
    ckbtn_dry["state"] = "enabled"
    ckbtn_force["state"] = "enabled"

    PROCESS.stop()
    PROCESS = None
    RUNNING = False

def stop_processing():
    global RUNNING, PROCESS
    btn_process_or_pause["text"] = "process"
    btn_stop["state"] = "disabled"
    btn_select_folder["state"] = "enabled"

    ckbtn_recursive["state"] = "enabled"
    ckbtn_dry["state"] = "enabled"
    ckbtn_force["state"] = "enabled"

    PROCESS.stop()
    progress_var.set(0)
    

def pause_processing():
    global RUNNING, PROCESS
    btn_process_or_pause["text"] = "process"
    RUNNING = False
    btn_stop["state"] = "disabled"
    btn_select_folder["state"] = "enabled"

    PROCESS.pause()

def continue_processing():
    global RUNNING, PROCESS
    btn_process_or_pause["text"] = "pause"
    RUNNING = True
    btn_stop["state"] = "enabled"
    btn_select_folder["state"] = "disabled"

    PROCESS.pause()


## button commands ##

def btn_fn_stop():
    global RUNNING

    if RUNNING:
        stop_processing()
        
def btn_fn_process():
    global RUNNING

    if RUNNING is False:
        if PROCESS is None:
            start_processing()
        else:
            continue_processing()

    else: # running
        pause_processing()

def btn_fn_select():
    global RUNNING, PROCESS
    filename = filedialog.askdirectory()

    if os.path.isdir(filename):
        folder_path.set(filename)
        btn_process_or_pause["state"] = "enabled"
        txt_dirpath.insert(END, filename)

    progress_var.set(0)
    PROCESS = None
    RUNNING = False

## define ui ##

if __name__ == "__main__":
        

    folder_path = StringVar(CONTENT)
    btn_select_folder = Button(CONTENT, text="select", command=btn_fn_select)

    btn_process_or_pause = Button(CONTENT, text="process", command=btn_fn_process)
    btn_process_or_pause["state"] = "disabled"

    btn_stop = Button(CONTENT, text="stop", command=btn_fn_stop)
    btn_stop["state"] = "disabled"

    txt_dirpath = Text(CONTENT, height=1, width=36) 

    progress_var = DoubleVar(CONTENT) 
    pbar_progress = Progressbar(CONTENT, orient=HORIZONTAL, variable=progress_var, length=300, max=100)


    count_var = StringVar(CONTENT)

    label_count = Label(CONTENT, textvariable=count_var) 


    CONTENT.grid(column=0, row=0)

    bvar_recursive = BooleanVar(CONTENT, value=True)
    ckbtn_recursive = Checkbutton(CONTENT, text="recurse", variable=bvar_recursive, onvalue=True)

    bvar_dry = BooleanVar(CONTENT, value=False)
    ckbtn_dry = Checkbutton(CONTENT, text="dry run", variable=bvar_dry, onvalue=True)

    bvar_force = BooleanVar(CONTENT, value=False)
    ckbtn_force = Checkbutton(CONTENT, text="force", variable=bvar_force, onvalue=True)





    ## place widgets on grid ##

    btn_select_folder.grid(column=0, row=1)
    btn_process_or_pause.grid(column=0, row=2)
    btn_stop.grid(column=0, row=3)

    ckbtn_recursive.grid(column=3, row=1)
    ckbtn_dry.grid(column=3, row=2)
    ckbtn_force.grid(column=3, row=3)

    txt_dirpath.grid(column=1, row=1)
    pbar_progress.grid(column=1, row=2)
    label_count.grid(column=1, row=3)



    ROOT.mainloop()
