DATE=2023_05_24

for BALL in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
python3 ./image_cutter.py --input_video_path=/home/iscander/Datasets/source/$DATE/ball${BALL}.avi --skip_n_frames=0 --output_path=/home/iscander/Datasets/cutout_images_${DATE}/ball${BALL}.pb

done
