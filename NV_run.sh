# NV cluster script

# source /home/ymingli/miniconda3/bin/activate
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate difix3d

# Force to shield site-packages
export PYTHONNOUSERSITE=1
# Clear PYTHONPATH and  LD_LIBRARY_PATH
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu


cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/Difix3D

python batched_process_wo_ref.py

# python render_video_difix_horizon_sine.py /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/ --fps 15

# rclone copy /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames3000_with_multi_frame_depth/spatial05_frames3000_with_multi_frame_depth_horizontal_sine_video_difix_wo_ref.mp4  "xiangyuDrive:Research/CityGS/RenderVideos/" -P