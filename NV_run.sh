# NV cluster script

# source /home/ymingli/miniconda3/bin/activate
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate difix3d

# Force to shield site-packages
export PYTHONNOUSERSITE=1
# Clear PYTHONPATH and  LD_LIBRARY_PATH
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu
video_output_path=/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4_parallel_left_difix3d_w_ref.mp4

cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/Difix3D

# python batched_process_w_ref.py \
#   --input_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/parallel_left/ours_199993/renders \
#   --ref_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/train/ours_199993/gt \
#   --output_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/parallel_left/ours_199993/renders_difix_w_ref \
#   --prompt "remove degradation"

# python batched_process_w_ref.py \
#   --input_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/parallel_right/ours_199993/renders \
#   --ref_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/train/ours_199993/gt \
#   --output_folder /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/parallel_right/ours_199993/renders_difix_w_ref \
#   --prompt "remove degradation"

python render_video_difix_parallel.py /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth_cap3M_L1depthReg05_randomBG_PostOptlr1e4/ --fps 15

rclone copy "${video_output_path}"  "xiangyuDrive:Research/CityGS/RenderVideos/" -P
# rclone copy /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/spatial05_frames3000_with_multi_frame_depth_horizontal_sine_video_difix_wo_ref.mp4  "xiangyuDrive:Research/CityGS/RenderVideos/" -P



# #### Dist
# python batched_process_w_ref_dist.py \
#   --input_folder /path/to/renders \
#   --ref_folder /path/to/gt \
#   --output_folder /path/to/renders_difix_w_ref \
#   --prompt "remove degradation" \
#   --num_gpus 8
