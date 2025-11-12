#include "glomap/estimators/relpose_estimation.h"

#include <colmap/util/threading.h>

#include <PoseLib/robust.h>

namespace glomap {

/**
 * [功能描述]：估计所有有效图像对的相对位姿（旋转R和平移t）
 * 使用PoseLib库结合RANSAC算法，从特征匹配中鲁棒地估计两视图之间的相对变换
 * @param view_graph：视图图，包含所有图像对及其特征匹配
 * @param cameras：相机内参集合，包含焦距、畸变等参数
 * @param images：图像集合，包含特征点坐标
 * @param options：相对位姿估计的配置选项，包含RANSAC参数和BA参数
 */
void EstimateRelativePoses(ViewGraph& view_graph,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<image_t, Image>& images,
                           const RelativePoseEstimationOptions& options) {
  // ========== 步骤1：收集所有有效的图像对 ==========
  std::vector<image_pair_t> valid_pair_ids;
  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;  // 跳过无效的图像对
    valid_pair_ids.push_back(image_pair_id);
  }

  // ========== 步骤2：准备分块并行处理 ==========
  const int64_t num_image_pairs = valid_pair_ids.size();
  const int64_t kNumChunks = 10;  // 将任务分成10个块，用于进度显示
  const int64_t interval =
      std::ceil(static_cast<double>(num_image_pairs) / kNumChunks);

  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);

  // ========== 步骤3：分块处理所有图像对 ==========
  LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";
  for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
    // 显示进度百分比
    std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%"
              << std::flush;
    const int64_t start = chunk_id * interval;
    const int64_t end =
        std::min<int64_t>((chunk_id + 1) * interval, num_image_pairs);

    // 为当前块中的每个图像对创建估计任务
    for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
      thread_pool.AddTask([&, pair_idx]() {
        // 定义为线程局部变量，在不同任务间复用内存分配，提高效率
        thread_local std::vector<Eigen::Vector2d> points2D_1;  // 图像1的2D点
        thread_local std::vector<Eigen::Vector2d> points2D_2;  // 图像2的2D点
        thread_local std::vector<char> inliers;                // 内点标记

        // 获取当前图像对及其相关数据
        ImagePair& image_pair =
            view_graph.image_pairs[valid_pair_ids[pair_idx]];
        const Image& image1 = images[image_pair.image_id1];
        const Image& image2 = images[image_pair.image_id2];
        const Eigen::MatrixXi& matches = image_pair.matches;  // 特征匹配关系，每行[idx1, idx2]

        // 获取相机内参并转换为PoseLib格式
        const Camera& camera1 = cameras[image1.camera_id];
        const Camera& camera2 = cameras[image2.camera_id];
        poselib::Camera camera_poselib1 = ColmapCameraToPoseLibCamera(camera1);
        poselib::Camera camera_poselib2 = ColmapCameraToPoseLibCamera(camera2);
        // 检查相机模型是否被PoseLib支持
        bool valid_camera_model =
            (camera_poselib1.model_id >= 0 && camera_poselib2.model_id >= 0);

        // ===== 子步骤1：收集匹配的2D点坐标 =====
        points2D_1.clear();
        points2D_2.clear();
        for (size_t idx = 0; idx < matches.rows(); idx++) {
          points2D_1.push_back(image1.features[matches(idx, 0)]);
          points2D_2.push_back(image2.features[matches(idx, 1)]);
        }
        
        // ===== 子步骤2：处理PoseLib不支持的相机模型 =====
        if (!valid_camera_model) {
          // 对于不支持的相机模型，需要手动去畸变并简化为针孔模型
          // 注意：这里仍然用焦距重新缩放，以保持RANSAC阈值不变
          Eigen::Matrix2d K1_new = Eigen::Matrix2d::Zero();
          Eigen::Matrix2d K2_new = Eigen::Matrix2d::Zero();
          K1_new(0, 0) = camera1.FocalLengthX();  // fx
          K1_new(1, 1) = camera1.FocalLengthY();  // fy
          K2_new(0, 0) = camera2.FocalLengthX();
          K2_new(1, 1) = camera2.FocalLengthY();
          
          // 对每个匹配点进行去畸变，然后用焦距重新缩放
          for (size_t idx = 0; idx < matches.rows(); idx++) {
            // 去畸变：畸变像素坐标 -> 归一化相机坐标 -> 用焦距缩放
            points2D_1[idx] = K1_new * camera1.CamFromImg(points2D_1[idx])
                                           .value_or(Eigen::Vector2d::Zero());
            points2D_2[idx] = K2_new * camera2.CamFromImg(points2D_2[idx])
                                           .value_or(Eigen::Vector2d::Zero());
          }

          // 将相机重置为简化的针孔模型（原焦距，零主点）
          // 这样PoseLib可以处理，且保持了合理的误差阈值
          camera_poselib1 = poselib::Camera(
              "PINHOLE",
              {camera1.FocalLengthX(), camera1.FocalLengthY(), 0., 0.},
              camera1.width,
              camera1.height);
          camera_poselib2 = poselib::Camera(
              "PINHOLE",
              {camera2.FocalLengthX(), camera2.FocalLengthY(), 0., 0.},
              camera2.width,
              camera2.height);
        }
        
        // ===== 子步骤3：使用RANSAC估计相对位姿 =====
        inliers.clear();
        poselib::CameraPose pose_rel_calc;  // 存储估计的相对位姿
        try {
          // PoseLib的核心估计函数
          // 算法流程：
          // 1. RANSAC循环：随机采样最小点集（5点算法）估计本质矩阵E
          // 2. 从E分解出相对位姿（R, t），选择三角化点在两相机前方的解
          // 3. 统计内点数量，保留最佳模型
          // 4. 使用所有内点进行Bundle Adjustment精化
          poselib::estimate_relative_pose(points2D_1,
                                          points2D_2,
                                          camera_poselib1,
                                          camera_poselib2,
                                          options.ransac_options,   // RANSAC参数（阈值、迭代次数等）
                                          options.bundle_options,   // BA优化参数
                                          &pose_rel_calc,           // 输出：相对位姿
                                          &inliers);                // 输出：内点标记
        } catch (const std::exception& e) {
          // 估计失败（如匹配点太少、退化配置等），标记为无效
          LOG(ERROR) << "Error in relative pose estimation: " << e.what();
          image_pair.is_valid = false;
          return;
        }

        // ===== 子步骤4：将PoseLib格式的位姿转换为GLOMAP格式 =====
        // 转换四元数表示（PoseLib和Eigen的四元数排列顺序不同）
        // PoseLib: [w, x, y, z], Eigen: [x, y, z, w]
        for (int i = 0; i < 4; i++) {
          image_pair.cam2_from_cam1.rotation.coeffs()[i] =
              pose_rel_calc.q[(i + 1) % 4];  // 循环移位调整顺序
        }
        // 复制平移向量
        image_pair.cam2_from_cam1.translation = pose_rel_calc.t;
      });
    }

    // 等待当前块的所有任务完成再继续下一块（分块同步）
    thread_pool.Wait();
  }

  // ========== 步骤4：完成 ==========
  std::cout << "\r Estimating relative pose: 100%" << std::endl;
  LOG(INFO) << "Estimating relative pose done";
}

}  // namespace glomap
