import os
from pathlib import Path
from diffusers.utils import load_image
from src.pipeline_difix import DifixPipeline

def batch_process(
    input_folder: str,
    output_folder: str,
    prompt: str,
    num_inference_steps: int = 1,
    timesteps: list = [199],
    guidance_scale: float = 0.0,
    device: str = "cuda"
):
    """
    Batch process all images in `input_folder` using the DifixPipeline and save results to `output_folder`.
    """
    # Initialize pipeline once
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to(device)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    # Iterate over files in input_folder
    for img_path in Path(input_folder).iterdir():
        if img_path.suffix.lower() not in extensions:
            continue

        # Prepare output path and skip if already processed
        output_path = Path(output_folder) / img_path.name
        if output_path.exists():
            print(f"Skipping {img_path.name}: output already exists.")
            continue
        try:
            # Load input image
            img = load_image(str(img_path))

            # Run the pipeline
            result = pipe(
                prompt,
                image=img,
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
    INPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_with_multi_frame_depth_9Mpts_Iter200k/train/ours_99993/renders"
    OUTPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_with_multi_frame_depth_9Mpts_Iter200k/train/ours_99993/renders_difix_wo_ref"
    PROMPT = "remove degradation"

    batch_process(
        INPUT_DIR,
        OUTPUT_DIR,
        PROMPT,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
        device="cuda"
    )
