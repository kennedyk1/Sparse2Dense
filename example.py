import Sparse2Dense

summary_kitti = Sparse2Dense.generate(
    image_input="dataset_examples/KITTI/data_tracking_image_2/training/image_02/0000/",
    pointcloud_input="dataset_examples/KITTI/data_tracking_velodyne/training/velodyne/0000",
    calibration_yaml="dataset_examples/KITTI/calibration.yaml",
    output_folder="output_examples/KITTI",
    depth_mask_size=13,
    intensity_mask_size=13,
    depth=True,
    intensity=True,
    debug=True,
    ratio_threshold=0.15,
    jump_threshold=0.01,
    device="cuda",
)
print(summary_kitti)


# Example 2:
# Process the MID-3K example folders.
summary_MID_3K = Sparse2Dense.generate(
    image_input="dataset_examples/MID-3K/MID-3K-rgb/images/",
    pointcloud_input="dataset_examples/MID-3K/MID-3K-pcd/pcd/",
    calibration_yaml="dataset_examples/MID-3K/calibration.yaml",
    output_folder="output_examples/MID-3K",
    depth_mask_size=17,
    intensity_mask_size=17,
    depth=True,
    intensity=True,
    debug=True,
    ratio_threshold=0.15,
    jump_threshold=0.01,
    device="cuda",
)
print(summary_MID_3K)
