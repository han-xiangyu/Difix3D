import os
import argparse
from pathlib import Path
from diffusers.utils import load_image
from src.pipeline_difix import DifixPipeline

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import re

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# def build_tasks(input_folder: Path, ref_folder: Path, output_folder: Path):
#     tasks = []
#     for img_path in sorted(input_folder.iterdir()):
#         if img_path.suffix.lower() not in EXTS:
#             continue

#         # 找参照图（同 stem，任意合法后缀）
#         ref_img_path = None
#         for ext in EXTS:
#             cand = ref_folder / f"{img_path.stem}{ext}"
#             if cand.exists():
#                 ref_img_path = cand
#                 break
#         if ref_img_path is None:
#             print(f"[SKIP] Ref not found for {img_path.name}")
#             continue

#         out_path = output_folder / f"{img_path.stem}_difix3d{img_path.suffix}"
#         tasks.append((img_path, ref_img_path, out_path))
#     return tasks

def build_tasks(input_folder: Path, ref_folder: Path, output_folder: Path):
    """
    将 input_folder 中的 *_left / *_right 图，与 ref_folder 中“不带该后缀”的原图配对。
    例：
      input:  foo_left.png  -> ref: foo.png / foo.jpg / ...
              foo_right.jpg -> ref: foo.png / foo.jpg / ...
    输出：
      <stem>_left_difix3d.<ext> / <stem>_right_difix3d.<ext>
    """
    tasks = []
    for img_path in sorted(input_folder.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue

        # 去掉末尾的 _left / _right（大小写不敏感）
        stem = img_path.stem
        base_stem = re.sub(r'_(left|right)$', '', stem, flags=re.IGNORECASE)

        # 如果这张图本身没有 _left/_right 后缀，则通常不是这轮要修复的外插图，跳过
        # （若你也想处理原图，把下面这段 if 注释掉即可）
        if base_stem == stem:
            # 既不是 left 也不是 right
            continue

        # 在 ref_folder 里找“不带后缀”的原图
        ref_img_path = None
        for ext in EXTS:
            cand = ref_folder / f"{base_stem}{ext}"
            if cand.exists():
                ref_img_path = cand
                break
        if ref_img_path is None:
            print(f"[SKIP] Ref not found for {img_path.name} (expect {base_stem}.* in {ref_folder})")
            continue

        out_path = output_folder / f"{img_path.stem}_difix3d{img_path.suffix}"
        tasks.append((img_path, ref_img_path, out_path))
    return tasks

def worker(rank, device, task_list, model_id, prompt, num_inference_steps, timesteps, guidance_scale, use_bf16=True):
    # 绑定设备 & 限制可见
    torch.cuda.set_device(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]

    # 提升吞吐（对 A100 友好）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # dtype 自动选择
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
    try:
        pipe = DifixPipeline.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        )
    except Exception:
        pipe = DifixPipeline.from_pretrained(
            model_id, trust_remote_code=True
        )
    pipe.to(device)

    pbar = tqdm(
        total=len(task_list),
        position=rank,               # 每个GPU一行
        desc=f"GPU {rank}",
        leave=True,
        dynamic_ncols=True,          # 适配终端宽度
        mininterval=0.5,             # 刷新最小间隔
        smoothing=0.2,               # 速度估计平滑
        bar_format=(
            "{desc} |{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
        unit="img"
    )
    
    for img_path, ref_img_path, out_path in task_list:
        try:
            if out_path.exists():
                print(f"[GPU {rank}] Skip exists: {out_path.name}")
                pbar.update(1)
                continue

            img = load_image(str(img_path))
            ref_img = load_image(str(ref_img_path))

            result = pipe(
                prompt,
                image=img,
                ref_image=ref_img,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(str(out_path))
            print(f"[GPU {rank}] Saved: {out_path.name}")

        except Exception as e:
            print(f"[GPU {rank}] Fail: {img_path.name} -> {e}")
        finally:
            pbar.update(1)
    pbar.close()

def split_evenly(items, n):
    """尽量均匀切成 n 份，前 r 份各多 1 个"""
    n = max(1, n)
    L = len(items)
    k, r = divmod(L, n)
    out, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        out.append(items[start:end])
        start = end
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--ref_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[199])
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--model_id", type=str, default="nvidia/difix_ref")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    ref_folder = Path(args.ref_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(input_folder, ref_folder, output_folder)
    if not tasks:
        print("[INFO] No tasks to run.")
        return

    visible = torch.cuda.device_count()
    if visible == 0:
        raise RuntimeError("No CUDA device found.")
    num_gpus = min(args.num_gpus or visible, visible)

    shards = split_evenly(tasks, num_gpus)

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(num_gpus):
        shard = shards[rank]
        if not shard:
            continue
        device = f"cuda:{rank}"
        p = ctx.Process(
            target=worker,
            args=(
                rank, device, shard,
                args.model_id, args.prompt,
                args.num_inference_steps, args.timesteps, args.guidance_scale,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[DONE] All workers finished.")

if __name__ == "__main__":
    main()
