DATE=2023_06_01

python3 ./training_set_generator.py --background_videos_template=/home/iscander/Datasets/cutout_images_${DATE}/background*.avi --input_images_path=/home/iscander/Datasets/cutout_images_${DATE} --num_train_images=10000 --num_val_images=1000 --num_test_images=500 --output_dataset_path=/home/iscander/Datasets/SelfSupervised_${DATE} --randomize_num_balls=True

