#include "glomap/processors/relpose_filter.h"

#include "glomap/math/rigid3d.h"

namespace glomap {

/**
 * [功能描述]：根据旋转误差过滤图像对，移除与全局旋转不一致的图像对
 * 通过比较从全局旋转计算的相对旋转与估计的相对旋转之间的角度差来检测异常值
 * @param view_graph：视图图，包含图像对及其相对位姿
 * @param images：图像集合，包含全局旋转信息（CamFromWorld）
 * @param max_angle：最大允许的旋转角度误差（度），超过此值的图像对将被标记为无效
 */
void RelPoseFilter::FilterRotations(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    double max_angle) {
  int num_invalid = 0;  // 统计被过滤掉的图像对数量
  
  // 遍历所有图像对
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 跳过已经被标记为无效的图像对
    if (image_pair.is_valid == false) continue;

    const Image& image1 = images.at(image_pair.image_id1);
    const Image& image2 = images.at(image_pair.image_id2);

    // 跳过未注册（未估计全局旋转）的图像
    if (image1.IsRegistered() == false || image2.IsRegistered() == false) {
      continue;
    }

    // ===== 计算从全局旋转推导的相对位姿 =====
    // cam2_from_cam1_calc = cam2_from_world * world_from_cam1
    //                     = cam2_from_world * (cam1_from_world)^(-1)
    // 这是从已估计的全局旋转计算出的相对位姿（理论值）
    Rigid3d pose_calc = image2.CamFromWorld() * Inverse(image1.CamFromWorld());

    // ===== 计算旋转角度误差 =====
    // 比较两个相对位姿的旋转差异：
    // 1. pose_calc：从全局旋转计算的相对位姿（旋转平均的结果）
    // 2. image_pair.cam2_from_cam1：从特征匹配估计的相对位姿（初始值）
    // 
    // 如果两者差异大，说明：
    // - 初始的相对位姿估计有误（误匹配导致）
    // - 或者该图像对与全局一致性模型不符（异常值）
    double angle = CalcAngle(pose_calc, image_pair.cam2_from_cam1);
    
    // 如果角度误差超过阈值，标记为无效
    if (angle > max_angle) {
      image_pair.is_valid = false;
      num_invalid++;
    }
  }

  LOG(INFO) << "Filtered " << num_invalid << " relative rotation with angle > "
            << max_angle << " degrees";
}

/**
 * [功能描述]：根据内点数量过滤图像对，移除内点数过少的图像对
 * 内点数太少说明特征匹配质量差或相对位姿估计不可靠，应该剔除
 * @param view_graph：视图图，包含所有图像对及其内点信息
 * @param min_inlier_num：最小内点数量阈值，低于此值的图像对将被标记为无效
 */
void RelPoseFilter::FilterInlierNum(ViewGraph& view_graph, int min_inlier_num) {
  int num_invalid = 0;  // 统计被过滤掉的图像对数量
  
  // 遍历所有图像对
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 跳过已经被标记为无效的图像对
    if (image_pair.is_valid == false) continue;

    // 检查内点数量是否满足最小要求
    // 内点数过少表明：
    // 1. 特征匹配质量差（误匹配多）
    // 2. 相对位姿估计不可靠（几何约束弱）
    // 3. 场景退化（如纯旋转、平面场景等）
    if (image_pair.inliers.size() < min_inlier_num) {
      image_pair.is_valid = false;  // 标记为无效
      num_invalid++;
    }
  }

  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier number < " << min_inlier_num;
}

/**
 * [功能描述]：根据内点比例过滤图像对，移除内点比例过低的图像对
 * 内点比例 = 内点数量 / 总匹配数量，相比绝对数量，比例更能反映匹配质量
 * @param view_graph：视图图，包含所有图像对及其内点信息
 * @param min_inlier_ratio：最小内点比例阈值（0-1之间），低于此值的图像对将被标记为无效
 */
void RelPoseFilter::FilterInlierRatio(ViewGraph& view_graph,
                                      double min_inlier_ratio) {
  int num_invalid = 0;  // 统计被过滤掉的图像对数量
  
  // 遍历所有图像对
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // 跳过已经被标记为无效的图像对
    if (image_pair.is_valid == false) continue;

    // 计算内点比例并判断是否满足最小要求
    // 内点比例 = 内点数量 / 总匹配数量
    // 例如：100个匹配中有30个内点，比例为0.3
    // 
    // 为什么使用比例而不是绝对数量：
    // 1. 更公平：200个匹配40个内点(20%)比100个匹配30个内点(30%)质量差
    // 2. 更鲁棒：不受图像特征数量影响
    // 3. 更能反映匹配质量：高比例说明误匹配少，几何约束可靠
    if (image_pair.inliers.size() / double(image_pair.matches.rows()) <
        min_inlier_ratio) {
      image_pair.is_valid = false;  // 标记为无效
      num_invalid++;
    }
  }

  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier ratio < " << min_inlier_ratio;
}

}  // namespace glomap