'''
def calculate_surface_distance(gt_img_path, pred_img_path):
    # load images
    gt_img = sitk.ReadImage(gt_img_path)
    pred_img = sitk.ReadImage(pred_img_path)

    # calculate surface distance
    gt_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True))
    pred_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))

    gt_surface = sitk.LabelContour(gt_img)
    pred_surface = sitk.LabelContour(pred_img)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(gt_surface)
    num_gt_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(pred_surface)
    num_pred_surface_pixels = int(statistics_image_filter.GetSum())

    gt2pred_dist_map =pred_dist_map*sitk.Cast(gt_surface, sitk.sitkFloat32)
    pred2gt_dist_map =gt_dist_map*sitk.Cast(pred_surface, sitk.sitkFloat32)
    gt2pred_dist_map_arr = sitk.GetArrayViewFromImage(gt2pred_dist_map)
    pred2gt_dist_map_arr = sitk.GetArrayViewFromImage(pred2gt_dist_map)

    gt2pred_dist = list(gt2pred_dist_map_arr[gt2pred_dist_map_arr!=0])
    gt2pred_dist = gt2pred_dist + list(np.zeros(num_gt_surface_pixels - len(gt2pred_dist)))

    pred2gt_dist = list(pred2gt_dist_map_arr[pred2gt_dist_map_arr!=0])
    pred2gt_dist = pred2gt_dist + list(np.zeros(num_pred_surface_pixels - len(pred2gt_dist)))

    all_dist = pred2gt_dist + gt2pred_dist
    mean_surface_dist = np.mean(all_dist)
    max_surface_dist = np.max(all_dist)
    median_surface_dist = np.median(all_dist)
    std_surface_dist = np.std(all_dist)


    print("######xxxxxx######")
    print(mean_surface_dist)
    print(max_surface_dist)
    print(median_surface_dist)
    print(std_surface_dist)



    print("mean surface distance {}".format(mean_surface_dist))

    return mean_surface_dist
'''