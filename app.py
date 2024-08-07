import time 
import torch
import argparse

from easyanimate.api.api import infer_forward_api, update_diffusion_transformer_api, update_edition_api
from easyanimate.ui.ui import ui_modelscope, ui_eas, ui, ui_casdao

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./outputs", type=str, help="The path to the output folder")
    parser.add_argument("--port", default=7860, type=int, help="The port to run the Gradio App on.")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="The host to run the Gradio App on.")
    parser.add_argument("--share", action="store_true", help="Whether to share this gradio demo.")
    parser.add_argument("--model_path", default="./models", type=str, help="The path to the models folder")
    parser.add_argument(
        "--low_gpu_memory_mode",
        action="store_true",
        help="Whether to enable low GPU memory mode",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "casdao"
    
    args = parse_args()
    
    # Low gpu memory mode, this is used when the GPU memory is under 16GB
    low_gpu_memory_mode = args.low_gpu_memory_mode
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = args.host
    server_port = args.port

    # Params below is used when ui_mode = "modelscope"
    edition = "v3"
    config_path = "config/easyanimate_video_slicevae_motion_module_v3.yaml"
    model_name = args.model_path
    savedir_sample = args.output

    if ui_mode == "modelscope":
        demo, controller = ui_modelscope(edition, config_path, model_name, savedir_sample, low_gpu_memory_mode, weight_dtype)
    elif ui_mode == "eas":
        demo, controller = ui_eas(edition, config_path, model_name, savedir_sample)
    elif ui_mode == "casdao":
        demo, controller = ui_casdao(model_name,savedir_sample, low_gpu_memory_mode, weight_dtype)
    else:
        demo, controller = ui(low_gpu_memory_mode, weight_dtype)

    # launch gradio
    app, _, _ = demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True,
        share=args.share
    )
    
    # launch api
    infer_forward_api(None, app, controller)
    update_diffusion_transformer_api(None, app, controller)
    update_edition_api(None, app, controller)
    
    # not close the python
    while True:
        time.sleep(5)