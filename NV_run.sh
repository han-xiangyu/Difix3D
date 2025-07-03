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

python render_video_difix.py /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_with_multi_frame_depth_9Mpts_Iter200k --fps 15

rclone copy /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial05_frames6000_with_multi_frame_depth_9Mpts_Iter200k/spatial05_frames6000_with_multi_frame_depth_9Mpts_Iter200k_train_set_video_difix_wo_ref.mp4  "xiangyuDrive:Research/CityGS/RenderVideos/" -P