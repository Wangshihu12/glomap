#include "view_graph_manipulation.h"

#include "glomap/math/two_view_geometry.h"
#include "glomap/math/union_find.h"

#include <colmap/util/threading.h>

namespace glomap {

image_pair_t ViewGraphManipulater::SparsifyGraph(
    ViewGraph& view_graph,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    int expected_degree) {
  image_t num_img = view_graph.KeepLargestConnectedComponents(frames, images);

  // Keep track of chosen edges
  std::unordered_set<image_pair_t> chosen_edges;
  const std::unordered_map<image_t, std::unordered_set<image_t>>&
      adjacency_list = view_graph.GetAdjacencyList();

  // Here, the average is the mean of the degrees
  double average_degree = 0;
  for (const auto& [image_id, neighbors] : adjacency_list) {
    if (images[image_id].IsRegistered() == false) continue;
    average_degree += neighbors.size();
  }
  average_degree = average_degree / num_img;

  // Go through the adjacency list and keep edge with probability
  // ((expected_degree * average_degree) / (degree1 * degree2))
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    image_t image_id1 = image_pair.image_id1;
    image_t image_id2 = image_pair.image_id2;

    if (images[image_id1].IsRegistered() == false ||
        images[image_id2].IsRegistered() == false)
      continue;

    int degree1 = adjacency_list.at(image_id1).size();
    int degree2 = adjacency_list.at(image_id2).size();

    if (degree1 <= expected_degree || degree2 <= expected_degree) {
      chosen_edges.insert(pair_id);
      continue;
    }

    if (rand() / double(RAND_MAX) <
        (expected_degree * average_degree) / (degree1 * degree2)) {
      chosen_edges.insert(pair_id);
    }
  }

  // Set all pairs not in the chosen edges to invalid
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (chosen_edges.find(pair_id) == chosen_edges.end()) {
      image_pair.is_valid = false;
    }
  }

  // Keep the largest connected component
  view_graph.KeepLargestConnectedComponents(frames, images);

  return chosen_edges.size();
}

image_t ViewGraphManipulater::EstablishStrongClusters(
    ViewGraph& view_graph,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    StrongClusterCriteria criteria,
    double min_thres,
    int min_num_images) {
  image_t num_img_before =
      view_graph.KeepLargestConnectedComponents(frames, images);

  // Construct the initial cluster by keeping the pairs with weight > min_thres
  UnionFind<image_pair_t> uf;
  // Go through the edges, and add the edge with weight > min_thres
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    bool status = false;
    status = status ||
             (criteria == INLIER_NUM && image_pair.inliers.size() > min_thres);
    status = status || (criteria == WEIGHT && image_pair.weight > min_thres);
    if (status) {
      uf.Union(image_pair_t(images[image_pair.image_id1].frame_id),
               image_pair_t(images[image_pair.image_id2].frame_id));
    }
  }

  // For every two connected components, we check the number of slightly weaker
  // pairs (> 0.75 min_thres) between them Two clusters are concatenated if the
  // number of such pairs is larger than a threshold (2)
  bool status = true;
  int iteration = 0;
  while (status) {
    status = false;
    iteration++;

    if (iteration > 10) {
      break;
    }

    std::unordered_map<image_pair_t, std::unordered_map<image_pair_t, int>>
        num_pairs;
    for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (image_pair.is_valid == false) continue;

      // If the number of inliers < 0.75 of the threshold, skip
      bool status = false;
      status = status || (criteria == INLIER_NUM &&
                          image_pair.inliers.size() < 0.75 * min_thres);
      status = status ||
               (criteria == WEIGHT && image_pair.weight < 0.75 * min_thres);
      if (status) continue;

      image_t image_id1 = image_pair.image_id1;
      image_t image_id2 = image_pair.image_id2;

      image_pair_t root1 = uf.Find(image_pair_t(images[image_id1].frame_id));
      image_pair_t root2 = uf.Find(image_pair_t(images[image_id2].frame_id));

      if (root1 == root2) {
        continue;
      }
      if (num_pairs.find(root1) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root1, std::unordered_map<image_pair_t, int>()));
      if (num_pairs.find(root2) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root2, std::unordered_map<image_pair_t, int>()));

      num_pairs[root1][root2]++;
      num_pairs[root2][root1]++;
    }
    // Connect the clusters progressively. If two clusters have more than 3
    // pairs, then connect them
    for (auto& [root1, counter] : num_pairs) {
      for (auto& [root2, count] : counter) {
        if (root1 <= root2) continue;

        if (count >= 2) {
          status = true;
          uf.Union(root1, root2);
        }
      }
    }
  }

  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    image_t image_id1 = image_pair.image_id1;
    image_t image_id2 = image_pair.image_id2;

    frame_t frame_id1 = images[image_id1].frame_id;
    frame_t frame_id2 = images[image_id2].frame_id;

    if (uf.Find(image_pair_t(frame_id1)) != uf.Find(image_pair_t(frame_id2))) {
      image_pair.is_valid = false;
    }
  }
  int num_comp = view_graph.MarkConnectedComponents(frames, images);

  LOG(INFO) << "Clustering take " << iteration << " iterations. "
            << "Images are grouped into " << num_comp
            << " clusters after strong-clustering";

  return num_comp;
}

/**
 * [功能描述]：更新图像对的配置类型（CALIBRATED/UNCALIBRATED）
 * 根据相机内参的可靠性，将未标定配置升级为标定配置，以提高相对位姿估计精度
 * @param view_graph：视图图，包含所有图像对及其几何关系
 * @param cameras：相机内参集合，包含焦距、畸变等参数
 * @param images：图像集合，每个图像关联一个相机
 */
void ViewGraphManipulater::UpdateImagePairsConfig(
    ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  // ========== 第一步：统计每个相机的配置情况 ==========
  // 对于每个相机，统计其参与的图像对的配置类型
  // pair.first：相机参与的图像对总数
  // pair.second：相机参与的CALIBRATED配置（本质矩阵）的图像对数量
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 跳过无效的图像对
    if (image_pair.is_valid == false) continue;

    // 获取图像对中两个图像对应的相机ID
    camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
    camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;

    // 获取相机对象
    const Camera& camera1 = cameras.at(camera_id1);
    const Camera& camera2 = cameras.at(camera_id2);
    // 如果相机没有先验焦距信息，跳过（无法判断其可靠性）
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length)
      continue;

    // 统计CALIBRATED配置（使用本质矩阵，内参已知）
    if (image_pair.config == colmap::TwoViewGeometry::CALIBRATED) {
      camera_counter[camera_id1].first++;   // 总数+1
      camera_counter[camera_id2].first++;
      camera_counter[camera_id1].second++;  // CALIBRATED数量+1
      camera_counter[camera_id2].second++;
    } 
    // 统计UNCALIBRATED配置（使用基础矩阵，内参未知或不可靠）
    else if (image_pair.config == colmap::TwoViewGeometry::UNCALIBRATED) {
      camera_counter[camera_id1].first++;   // 仅总数+1
      camera_counter[camera_id2].first++;
    }
  }

  // ========== 第二步：判断每个相机的有效性 ==========
  // 如果一个相机在大多数（>50%）图像对中使用CALIBRATED配置，
  // 则认为该相机的内参是可靠的
  std::unordered_map<camera_t, bool> camera_validity;
  for (auto& [camera_id, counter] : camera_counter) {
    if (counter.first == 0) {
      // 相机未参与任何图像对，标记为无效
      camera_validity[camera_id] = false;
    } else if (counter.second * 1. / counter.first > 0.5) {
      // CALIBRATED配置的比例超过50%，相机内参可靠
      camera_validity[camera_id] = true;
    } else {
      // CALIBRATED配置的比例不足50%，相机内参不可靠
      camera_validity[camera_id] = false;
    }
  }

  // ========== 第三步：更新图像对的配置 ==========
  // 将两个相机都可靠的UNCALIBRATED图像对升级为CALIBRATED
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 只处理有效且当前为UNCALIBRATED的图像对
    if (image_pair.is_valid == false) continue;
    if (image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED) continue;

    // 获取图像对对应的相机ID
    camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
    camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;

    // 获取相机对象
    const Camera& camera1 =
        cameras.at(images.at(image_pair.image_id1).camera_id);
    const Camera& camera2 =
        cameras.at(images.at(image_pair.image_id2).camera_id);

    // 如果两个相机的内参都可靠，则升级配置
    if (camera_validity[camera_id1] && camera_validity[camera_id2]) {
      // 将配置从UNCALIBRATED（基础矩阵）升级为CALIBRATED（本质矩阵）
      image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
      // 根据相机内参和相对运动重新计算基础矩阵F
      // F = K2^(-T) * E * K1^(-1)，其中E是本质矩阵
      FundamentalFromMotionAndCameras(
          camera1, camera2, image_pair.cam2_from_cam1, &image_pair.F);
    }
  }
}

// Decompose the relative camera postion from the camera config
/**
 * [功能描述]：从两视图几何矩阵（本质矩阵E、基础矩阵F、单应矩阵H）中分解出相对位姿
 * 即从矩阵形式恢复出旋转矩阵R和平移向量t，构建完整的相对变换
 * @param view_graph：视图图，包含所有图像对及其几何关系
 * @param cameras：相机内参集合，包含焦距、畸变等参数
 * @param images：图像集合，包含特征点等信息
 */
void ViewGraphManipulater::DecomposeRelPose(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images) {
  // ========== 第一步：收集需要分解的图像对 ==========
  std::vector<image_pair_t> image_pair_ids;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 跳过无效的图像对
    if (image_pair.is_valid == false) continue;
    // 只处理两个相机都有先验焦距的图像对
    // 因为相对位姿分解需要已知的相机内参
    if (!cameras[images[image_pair.image_id1].camera_id]
             .has_prior_focal_length ||
        !cameras[images[image_pair.image_id2].camera_id].has_prior_focal_length)
      continue;
    image_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = image_pair_ids.size();
  LOG(INFO) << "Decompose relative pose for " << num_image_pairs << " pairs";

  // ========== 第二步：并行分解相对位姿 ==========
  // 使用线程池加速处理，每个图像对的分解是独立的
  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);
  for (int64_t idx = 0; idx < num_image_pairs; idx++) {
    thread_pool.AddTask([&, idx]() {
      ImagePair& image_pair = view_graph.image_pairs.at(image_pair_ids[idx]);
      image_t image_id1 = image_pair.image_id1;
      image_t image_id2 = image_pair.image_id2;

      camera_t camera_id1 = images.at(image_id1).camera_id;
      camera_t camera_id2 = images.at(image_id2).camera_id;

      // 使用COLMAP的两视图几何结构重新估计相对位姿
      // 准备输入数据：E矩阵、F矩阵、H矩阵和配置类型
      colmap::TwoViewGeometry two_view_geometry;
      two_view_geometry.E = image_pair.E;  // 本质矩阵（Essential Matrix）
      two_view_geometry.F = image_pair.F;  // 基础矩阵（Fundamental Matrix）
      two_view_geometry.H = image_pair.H;  // 单应矩阵（Homography Matrix）
      two_view_geometry.config = image_pair.config;  // 配置类型

      // 从几何矩阵中分解出相对位姿（旋转R和平移t）
      // 使用特征点进行验证，选择最佳的分解结果（E矩阵有4个可能的分解）
      colmap::EstimateTwoViewGeometryPose(cameras[camera_id1],
                                          images[image_id1].features,
                                          cameras[camera_id2],
                                          images[image_id2].features,
                                          &two_view_geometry);

      // ===== 处理特殊情况 =====
      // 如果场景是平面的（PLANAR），且两个相机都有先验焦距
      // 则将配置升级为CALIBRATED（标定配置）
      if (image_pair.config == colmap::TwoViewGeometry::PLANAR &&
          cameras[camera_id1].has_prior_focal_length &&
          cameras[camera_id2].has_prior_focal_length) {
        image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
        return;
      } 
      // 如果任一相机没有先验焦距，无法可靠分解，跳过
      else if (!(cameras[camera_id1].has_prior_focal_length &&
                   cameras[camera_id2].has_prior_focal_length))
        return;

      // 更新图像对的配置和相对位姿
      image_pair.config = two_view_geometry.config;
      image_pair.cam2_from_cam1 = two_view_geometry.cam2_from_cam1;  // 相机2相对相机1的变换

      // 归一化平移向量（因为从E矩阵分解出的平移只有方向，没有尺度）
      // 统一设置为单位向量，实际尺度需要通过三角化等方法恢复
      if (image_pair.cam2_from_cam1.translation.norm() > EPS) {
        image_pair.cam2_from_cam1.translation =
            image_pair.cam2_from_cam1.translation.normalized();
      }
    });
  }

  // 等待所有线程完成
  thread_pool.Wait();

  // ========== 第三步：统计纯旋转的图像对 ==========
  // 纯旋转表示相机只旋转没有平移（如相机原地转动拍摄全景图）
  size_t counter = 0;
  for (size_t idx = 0; idx < image_pair_ids.size(); idx++) {
    ImagePair& image_pair = view_graph.image_pairs.at(image_pair_ids[idx]);
    // 如果配置既不是CALIBRATED也不是PLANAR_OR_PANORAMIC，说明是纯旋转
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC)
      counter++;
  }
  LOG(INFO) << "Decompose relative pose done. " << counter
            << " pairs are pure rotation";
}

}  // namespace glomap
