# Execute from a standalone env if conflict with the main one

import numpy as np
import rasterio
from skimage.measure import label
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error


def read_tif(file_path):
    """Reads a TIF file and returns the data as a numpy array."""
    with rasterio.open(file_path) as src:
        return src.read(1)  # Read the first band


def compute_median_height_per_building(height_map, building_mask, building_labels):
    """Computes the median height of each building."""
    building_medians = []
    for building_label in np.unique(building_labels):
        if building_label == 0:  # Skip the background
            continue
        # Get the indices of the current building
        building_indices = np.where(building_labels == building_label)
        # Extract the height values for the current building
        building_heights = height_map[building_indices]
        # Compute the median height for this building
        building_medians.append(np.median(building_heights))
    return np.array(building_medians)


def rmse(true_values, predicted_values):
    """Computes the RMSE between two sets of values."""
    return np.sqrt(mean_squared_error(true_values, predicted_values))


def evaluate_buildingwise_errors(pred_height_map_path, gt_height_map_path, building_mask_path):
    # Load the predicted, GT height maps and the building mask
    pred_height_map = read_tif(pred_height_map_path)
    gt_height_map = read_tif(gt_height_map_path)
    building_mask = read_tif(building_mask_path)

    # Label connected components in the building mask
    building_labels = label(building_mask, connectivity=2)

    # Compute the median height for each building (connected component)
    pred_medians = compute_median_height_per_building(pred_height_map, building_mask, building_labels)
    gt_medians = compute_median_height_per_building(gt_height_map, building_mask, building_labels)

    # Compute the building-wise RMSE (RMSE-B)
    buildingwise_rmse = rmse(gt_medians, pred_medians)

    # Compute the building-wise MAE
    buildingwise_mae = mean_absolute_error(gt_medians, pred_medians)

    # Compute the building-wise Median Absolute Error (MedAE)
    buildingwise_medae = median_absolute_error(gt_medians, pred_medians)

    return buildingwise_rmse, buildingwise_mae, buildingwise_medae


def evaluate_raster():
    # Evaluation based on raster

    # Munich-cloud
    pred_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_cloud_001600.tiff"
    gt_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_ndsm_chunk5.tif"
    building_mask_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_footprint_chunk5.tif"

    # Munich-image
    # pred_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_image_001300.tiff"
    # gt_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_ndsm_chunk5.tif"
    # building_mask_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_footprint_chunk5.tif"

    # Munich-cloud+image
    # pred_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_cloud+image_001200.tiff"
    # gt_height_map_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_ndsm_chunk5.tif"
    # building_mask_path = "/workspace/projects/tomosar2height/data/munich/raster/instance_eval/munich_footprint_chunk5.tif"

    # Berlin-cloud
    # pred_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_cloud_005700.tiff"
    # gt_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_ndsm_chunk11.tif"
    # building_mask_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_footprint_chunk11.tif"

    # Berlin-image
    # pred_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_image_007900.tiff"
    # gt_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_ndsm_chunk11.tif"
    # building_mask_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_footprint_chunk11.tif"

    # Berlin-cloud+image
    # pred_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_cloud+image_001500.tiff"
    # gt_height_map_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_ndsm_chunk11.tif"
    # building_mask_path = "/workspace/projects/tomosar2height/data/berlin/raster/instance_eval/berlin_footprint_chunk11.tif"


    rmse_b, mae_b, medae_b = evaluate_buildingwise_errors(pred_height_map_path, gt_height_map_path, building_mask_path)

    print(f"Building-wise RMSE (RMSE-B): {rmse_b}")
    print(f"Building-wise MAE (MAE-B): {mae_b}")
    print(f"Building-wise MedAE (MedAE-B): {medae_b}")

    # Berlin-cloud
    # Building-wise RMSE (RMSE-B): 6.170395851135254
    # Building-wise MAE (MAE-B): 3.6855392456054688
    # Building-wise MedAE (MedAE-B): 2.3181204795837402

    # Berlin-image
    # Building-wise RMSE (RMSE-B): 6.748867034912109
    # Building-wise MAE (MAE-B): 4.611605167388916
    # Building-wise MedAE (MedAE-B): 3.2409770488739014

    # Berlin-cloud+image
    # Building-wise RMSE (RMSE-B): 5.352768421173096
    # Building-wise MAE (MAE-B): 3.5444416999816895
    # Building-wise MedAE (MedAE-B): 2.5691750049591064

    # Munich-cloud
    # Building-wise RMSE (RMSE-B): 6.866394996643066
    # Building-wise MAE (MAE-B): 5.0606770515441895
    # Building-wise MedAE (MedAE-B): 3.314073085784912

    # Munich-image
    # Building-wise RMSE (RMSE-B): 4.830209255218506
    # Building-wise MAE (MAE-B): 3.4618468284606934
    # Building-wise MedAE (MedAE-B): 2.514556407928467

    # Munich-cloud+image
    # Building-wise RMSE (RMSE-B): 4.614895343780518
    # Building-wise MAE (MAE-B): 3.312502145767212
    # Building-wise MedAE (MedAE-B): 2.5021157264709473


def read_npz(file_path):
    """Reads an NPZ file and returns the points as a numpy array (x, y, z)."""
    data = np.load(file_path)
    # Assuming the keys for x, y, z are 'x', 'y', 'z'. Adjust if your NPZ keys are different.
    xyz = data['pts']
    return xyz


def associate_points_with_buildings(points, building_mask, building_labels, transform):
    """
    Associates each point in the point cloud with a building instance based on the building mask.

    Args:
        points (np.ndarray): Point cloud (N, 3) with x, y, z coordinates.
        building_mask (np.ndarray): Building mask raster.
        building_labels (np.ndarray): Labeled building mask raster.
        transform (Affine): Transformation matrix from rasterio.

    Returns:
        dict: Dictionary mapping building labels to point heights.
    """
    # Initialize a dictionary to hold building points
    building_points = {label: [] for label in np.unique(building_labels) if label != 0}

    # Transform points to raster indices
    raster_x, raster_y = (~transform) * (points[:, 0], points[:, 1])
    raster_x = np.clip(np.floor(raster_x).astype(int), 0, building_mask.shape[1] - 1)
    raster_y = np.clip(np.floor(raster_y).astype(int), 0, building_mask.shape[0] - 1)

    # Associate each point with a building
    for px, py, pz in zip(raster_x, raster_y, points[:, 2]):
        building_label = building_labels[py, px]
        if building_label > 0:
            building_points[building_label].append(pz)

    return {label: np.array(heights) for label, heights in building_points.items()}


def evaluate_cloud_valid_only():
    # Only on instances with valid point coverage

    # File paths
    point_cloud_path = "/workspace/projects/tomosar2height/data/munich/generated/chunk_005/input_point_cloud.npz"
    dtm_path = "/workspace/projects/tomosar2height/data/munich/raster/munich_chunk5_dem.tif"
    building_mask_path = "/workspace/projects/tomosar2height/data/munich/raster/munich_chunk5_mask.tif"
    ndsm_path = "/workspace/projects/tomosar2height/data/munich/raster/ndsm_chunk5.tif"

    # Read the point cloud, DTM, nDSM, and building mask
    points = read_npz(point_cloud_path)  # Load NPZ file
    dtm = read_tif(dtm_path)[:-1, :]
    ndsm = read_tif(ndsm_path)[:, :]
    building_mask = read_tif(building_mask_path)[1:-1, :]  # Adjust for consistency

    # Label connected components in the building mask
    from skimage.measure import label
    with rasterio.open(building_mask_path) as src:
        building_labels = label(building_mask, connectivity=2)
        transform = src.transform

    # Associate points with buildings
    building_points = associate_points_with_buildings(points, building_mask, building_labels, transform)

    # Compute the median height for each building from the point cloud
    pred_medians = []
    for label, heights in building_points.items():
        if heights.size > 0:  # Skip empty buildings
            pred_medians.append(np.median(heights))
        else:
            pred_medians.append(np.nan)  # Assign NaN for empty buildings

    # Compute the median height for each building from the DTM
    dtm_medians = compute_median_height_per_building(dtm, building_mask, building_labels)

    # Calculate the predicted height above DTM (predicted median - DTM median)
    pred_minus_dtm = np.array(pred_medians) - np.array(dtm_medians)

    # Compute the median height for each building from the nDSM
    ndsm_medians = compute_median_height_per_building(ndsm, building_mask, building_labels)

    # Remove NaN values from all arrays
    pred_minus_dtm = np.array(pred_minus_dtm)
    ndsm_medians = np.array(ndsm_medians)
    valid_indices = ~np.isnan(pred_minus_dtm) & ~np.isnan(ndsm_medians)

    pred_minus_dtm = pred_minus_dtm[valid_indices]
    ndsm_medians = ndsm_medians[valid_indices]

    # Compute metrics
    buildingwise_rmse = rmse(ndsm_medians, pred_minus_dtm)
    buildingwise_mae = mean_absolute_error(ndsm_medians, pred_minus_dtm)
    buildingwise_medae = median_absolute_error(ndsm_medians, pred_minus_dtm)

    print(f"Building-wise RMSE (RMSE-B): {buildingwise_rmse}")
    print(f"Building-wise MAE (MAE-B): {buildingwise_mae}")
    print(f"Building-wise MedAE (MedAE-B): {buildingwise_medae}")

    # Building-wise MAE (MAE-B): 5.8460189898324
    # Building-wise RMSE (RMSE-B): 7.133451927384065
    # Building-wise MedAE (MedAE-B): 5.113525418323775


def evaluate_cloud_all():
    # All instances regardless of point coverage

    # File paths
    point_cloud_path = "/workspace/projects/tomosar2height/data/munich/generated/chunk_005/input_point_cloud.npz"
    dtm_path = "/workspace/projects/tomosar2height/data/munich/raster/munich_chunk5_dem.tif"
    building_mask_path = "/workspace/projects/tomosar2height/data/munich/raster/munich_chunk5_mask.tif"
    ndsm_path = "/workspace/projects/tomosar2height/data/munich/raster/ndsm_chunk5.tif"

    # Read the point cloud, DTM, nDSM, and building mask
    points = read_npz(point_cloud_path)  # Load NPZ file
    dtm = read_tif(dtm_path)[:-1, :]
    ndsm = read_tif(ndsm_path)[:, :]
    building_mask = read_tif(building_mask_path)[1:-1, :]  # Adjust for consistency

    # Label connected components in the building mask
    from skimage.measure import label
    with rasterio.open(building_mask_path) as src:
        building_labels = label(building_mask, connectivity=2)
        transform = src.transform

    # Associate points with buildings
    building_points = associate_points_with_buildings(points, building_mask, building_labels, transform)

    # Compute the median height for each building from the point cloud
    pred_medians = []
    for label, heights in building_points.items():
        if heights.size > 0:  # Skip empty buildings
            pred_medians.append(np.median(heights))
        else:
            pred_medians.append(np.nan)  # Assign 0 for empty buildings

    # Compute the median height for each building from the DTM
    dtm_medians = compute_median_height_per_building(dtm, building_mask, building_labels)

    # Calculate the predicted height above DTM (predicted median - DTM median)
    pred_minus_dtm = np.array(pred_medians) - np.array(dtm_medians)

    # Compute the median height for each building from the nDSM
    ndsm_medians = compute_median_height_per_building(ndsm, building_mask, building_labels)

    # Replace NaN in pred_minus_dtm with 0 for all building instances
    pred_minus_dtm = np.nan_to_num(pred_minus_dtm)

    # Print predicted and reference heights for every building
    print("Building Instance Heights:")
    for i, (pred, ref) in enumerate(zip(pred_minus_dtm, ndsm_medians)):
        print(f"Building {i + 1}: Predicted Height = {pred:.2f}, Reference Height = {ref:.2f}")

    # Metrics can now include all building instances
    buildingwise_rmse = rmse(ndsm_medians, pred_minus_dtm)
    buildingwise_mae = mean_absolute_error(ndsm_medians, pred_minus_dtm)
    buildingwise_medae = median_absolute_error(ndsm_medians, pred_minus_dtm)

    print(f"Building-wise MAE (MAE-B): {buildingwise_mae}")
    print(f"Building-wise RMSE (RMSE-B): {buildingwise_rmse}")
    print(f"Building-wise MedAE (MedAE-B): {buildingwise_medae}")
    # Building-wise MAE (MAE-B): 5.249051827510731
    # Building-wise RMSE (RMSE-B): 6.680501389655927
    # Building-wise MedAE (MedAE-B): 4.369262741146827


if __name__ == '__main__':
    # evaluate_raster()
    # evaluate_cloud_valid_only()
    evaluate_cloud_all()
