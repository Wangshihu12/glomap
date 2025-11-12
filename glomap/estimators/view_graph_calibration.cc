#include "glomap/estimators/view_graph_calibration.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/math/two_view_geometry.h"

#include <colmap/scene/two_view_geometry.h>

#include <thread>

namespace glomap {

/**
 * [功能描述]：视图图校准的核心求解函数，通过优化相机内参提高重建精度
 * 使用Ceres非线性优化库，基于图像对之间的几何约束（本质矩阵/基础矩阵）来校准相机焦距等内参
 * @param view_graph：视图图，包含所有图像对及其几何关系（E、F、H矩阵等）
 * @param cameras：相机内参集合，将被优化更新
 * @param images：图像集合，包含特征点等信息
 * @return 返回优化是否成功：true表示求解可用，false表示求解失败
 */
bool ViewGraphCalibrator::Solve(ViewGraph& view_graph,
                                std::unordered_map<camera_t, Camera>& cameras,
                                std::unordered_map<image_t, Image>& images) {
  // ========== 步骤1：初始化优化问题 ==========
  LOG(INFO) << "Start ViewGraphCalibrator";

  // 重置优化问题，清空之前的残差块和参数块
  Reset(cameras);

  // ========== 步骤2：选择线性求解器类型 ==========
  // 根据相机数量选择合适的求解器，以平衡计算效率和内存使用
  if (cameras.size() < 50)
    // 相机数量较少时，使用稠密求解器（速度快但内存占用高）
    options_.solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  else
    // 相机数量较多时，使用稀疏求解器（内存效率高，适合大规模问题）
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  // ========== 步骤3：构建优化问题 ==========
  // 将图像对添加到优化问题中，每个图像对贡献一个或多个残差项
  // 残差来自几何约束：x2^T · F · x1 = 0（基础矩阵约束）或 p2^T · E · p1 = 0（本质矩阵约束）
  AddImagePairsToProblem(view_graph, cameras, images);

  // ========== 步骤4：参数化相机 ==========
  // 设置哪些相机参数需要优化，哪些保持不变
  // 如果相机有可靠的先验内参（has_prior_focal_length），则将其固定
  const size_t num_cameras = ParameterizeCameras(cameras);

  // 如果没有相机需要优化（所有相机都有可靠的先验内参），直接返回
  if (num_cameras == 0) {
    LOG(INFO) << "No cameras to optimize";
    return true;
  }

  // ========== 步骤5：执行非线性优化 ==========
  // 使用Ceres求解器最小化重投影误差，优化相机内参
  ceres::Solver::Summary summary;
  // 根据日志级别决定是否输出优化进度信息
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  // 调用Ceres求解器进行优化
  // 目标：min Σ (x2^T · F(K1, K2, R, t) · x1)^2
  // 优化变量：相机内参（焦距、主点、畸变系数等）
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  // 输出详细的优化报告（仅在日志级别为2时）
  VLOG(2) << summary.FullReport();

  // ========== 步骤6：后处理 ==========
  // 将优化后的参数复制回相机对象
  // 因为Ceres内部使用自己的参数表示，需要转换回原始数据结构
  CopyBackResults(cameras);
  
  // 根据优化后的内参过滤图像对
  // 移除与新内参不一致的图像对（可能是异常值）
  FilterImagePairs(view_graph);

  // 返回优化是否成功
  // IsSolutionUsable() 检查求解器是否收敛且结果可用
  return summary.IsSolutionUsable();
}

void ViewGraphCalibrator::Reset(
    const std::unordered_map<camera_t, Camera>& cameras) {
  // Initialize the problem
  focals_.clear();
  focals_.reserve(cameras.size());
  for (const auto& [camera_id, camera] : cameras) {
    focals_[camera_id] = camera.Focal();
  }

  // Set up the problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
}

void ViewGraphCalibrator::AddImagePairsToProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED)
      continue;
    if (image_pair.is_valid == false) continue;

    AddImagePair(image_pair, cameras, images);
  }
}

void ViewGraphCalibrator::AddImagePair(
    const ImagePair& image_pair,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  const camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
  const camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;

  if (camera_id1 == camera_id2) {
    problem_->AddResidualBlock(
        FetzerFocalLengthSameCameraCost::Create(
            image_pair.F, cameras.at(camera_id1).PrincipalPoint()),
        loss_function_.get(),
        &(focals_[camera_id1]));
  } else {
    problem_->AddResidualBlock(
        FetzerFocalLengthCost::Create(image_pair.F,
                                      cameras.at(camera_id1).PrincipalPoint(),
                                      cameras.at(camera_id2).PrincipalPoint()),
        loss_function_.get(),
        &(focals_[camera_id1]),
        &(focals_[camera_id2]));
  }
}

size_t ViewGraphCalibrator::ParameterizeCameras(
    const std::unordered_map<camera_t, Camera>& cameras) {
  size_t num_cameras = 0;
  for (auto& [camera_id, camera] : cameras) {
    if (!problem_->HasParameterBlock(&(focals_[camera_id]))) continue;

    num_cameras++;
    problem_->SetParameterLowerBound(&(focals_[camera_id]), 0, 1e-3);
    if (camera.has_prior_focal_length) {
      problem_->SetParameterBlockConstant(&(focals_[camera_id]));
      num_cameras--;
    }
  }

  return num_cameras;
}

void ViewGraphCalibrator::CopyBackResults(
    std::unordered_map<camera_t, Camera>& cameras) {
  size_t counter = 0;
  for (auto& [camera_id, camera] : cameras) {
    if (!problem_->HasParameterBlock(&(focals_[camera_id]))) continue;

    // if the estimated parameter is too crazy, reject it
    if (focals_[camera_id] / camera.Focal() > options_.thres_higher_ratio ||
        focals_[camera_id] / camera.Focal() < options_.thres_lower_ratio) {
      VLOG(2) << "Ignoring degenerate camera camera " << camera_id
              << " focal: " << focals_[camera_id]
              << " original focal: " << camera.Focal();
      counter++;

      continue;
    }

    // Marke that the camera has refined intrinsics
    camera.has_refined_focal_length = true;

    // Update the focal length
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] = focals_[camera_id];
    }
  }
  LOG(INFO) << counter << " cameras are rejected in view graph calibration";
}

size_t ViewGraphCalibrator::FilterImagePairs(ViewGraph& view_graph) const {
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = options_.solver_options.num_threads;
  eval_options.apply_loss_function = false;
  std::vector<double> residuals;
  problem_->Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

  // Dump the residuals into the original data structure
  size_t counter = 0;
  size_t invalid_counter = 0;

  const double thres_two_view_error_sq =
      options_.thres_two_view_error * options_.thres_two_view_error;

  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED)
      continue;
    if (image_pair.is_valid == false) continue;

    const Eigen::Vector2d error(residuals[counter], residuals[counter + 1]);

    // Set the two view geometry to be invalid if the error is too high
    if (error.squaredNorm() > thres_two_view_error_sq) {
      invalid_counter++;
      image_pair.is_valid = false;
    }

    counter += 2;
  }

  LOG(INFO) << "invalid / total number of two view geometry: "
            << invalid_counter << " / " << (counter / 2);

  return invalid_counter;
}

}  // namespace glomap
