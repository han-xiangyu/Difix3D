import os
from pathlib import Path
from diffusers.utils import load_image
from src.pipeline_difix import DifixPipeline

def batch_process(
    input_folder: str,
    ref_folder: str,
    output_folder: str,
    prompt: str,
    num_inference_steps: int = 1,
    timesteps: list = [199],
    guidance_scale: float = 0.0,
    device: str = "cuda"
):
    """
    Batch process all images in `input_folder` alongside corresponding reference images in `ref_folder` 
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

    
    # Iterate over files in input_folder
    for img_path in Path(input_folder).iterdir():
        if img_path.suffix.lower() not in extensions:
            continue

        # Prepare output path and skip if already processed
        output_path = Path(output_folder) / f"{img_path.stem}_difix3d{img_path.suffix}"
        if output_path.exists():
            print(f"Skipping {img_path.name}: output already exists.")
            continue
        try:

            # Locate matching reference image (same stem, any supported extension)
            ref_img = None
            for ext in extensions:
                candidate_ref_path = Path(ref_folder) / f"{img_path.stem}{ext}"
                if candidate_ref_path.exists():
                    ref_img = load_image(str(candidate_ref_path))
                    break

            if ref_img is None:
                print(f"Reference image not found for {img_path.stem}, skipping.")
                continue


            # Load input and reference images
            img = load_image(str(img_path))
            

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
            output_path = Path(output_folder) / f"{img_path.stem}_difix3d{img_path.suffix}"
            result.images[0].save(str(output_path))
            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    # Example usage
    INPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/parallel_right/ours_124993/renders"
    REF_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/train/ours_124993/gt"
    OUTPUT_DIR = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/parallel_right/ours_124993/renders_difix_w_ref"
    PROMPT = "remove degradation"

    batch_process(
        INPUT_DIR,
        REF_DIR,
        OUTPUT_DIR,
        PROMPT,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
        device="cuda"
    )
