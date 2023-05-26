DATE=2023_05_25

python3 ./training_set_generator.py --input_images_path=/home/iscander/Datasets/cutout_images_${DATE} --num_train_images=5000 --num_val_images=500 --num_test_images=300 --output_dataset_path=/home/iscander/Datasets/SelfSupervised_${DATE}
