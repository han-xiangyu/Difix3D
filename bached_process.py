import os
from pathlib import Path
from diffusers.utils import load_image
from src.pipeline_difix import DifixPipeline

def batch_process(
    input_folder: str,
    ref_path: str,
    output_folder: str,
    prompt: str,
    num_inference_steps: int = 1,
    timesteps: list = [199],
    guidance_scale: float = 0.0,
    device: str = "cuda"
):
    """
    Batch process all images in `input_folder` alongside corresponding reference images in `ref_path` 
    using the DifixPipeline, and save results to `output_folder`.

    Assumes reference images share the same filenames as input images.
    """
    # Initialize pipeline once
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to(device)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    ref_path = Path(ref_path)
    if not ref_path.exists():
        print(f"Reference image not found for {img_path.name}, skipping.")

    # Iterate over files in input_folder
    for img_path in Path(input_folder).iterdir():
        if img_path.suffix.lower() in extensions:

            try:
                # Load input and reference images
                img = load_image(str(img_path))
                ref_img = load_image(str(ref_path))

                # Run the pipeline with reference image
                result = pipe(
                    prompt,
                    image=img,
                    ref_image=ref_img,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale
                )

                # Save output image
                output_path = Path(output_folder) / img_path.name
                result.images[0].save(str(output_path))
                print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    # Example usage
    INPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_per_8M_pts/train/ours_99993/renders"
    REF_PATH = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_per_8M_pts/train/ours_99993/gt/trav_2_channel_1_img_1000.jpg"
    OUTPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_per_8M_pts/train/ours_99993/renders_difix_w_ref"
    PROMPT = "remove degradation"

    batch_process(
        INPUT_DIR,
        REF_PATH,
        OUTPUT_DIR,
        PROMPT,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
        device="cuda"
    )
