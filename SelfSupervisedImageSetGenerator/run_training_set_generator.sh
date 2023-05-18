DATE=2023_05_17

python3 ./training_set_generator.py --input_images_path=/home/iscander/Datasets/cutout_images_${DATE} --num_images_to_generate=100 --output_dataset_path=/home/iscander/Datasets/SelfSupervised_${DATE}
