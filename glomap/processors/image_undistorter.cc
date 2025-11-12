#include "glomap/processors/image_undistorter.h"

#include <colmap/util/threading.h>

namespace glomap {

/**
 * [功能描述]：对图像特征点进行去畸变处理，将畸变的像素坐标转换为归一化的相机坐标
 * 去畸变是SfM流水线中的关键预处理步骤，消除镜头畸变的影响，使后续的几何计算更准确
 * @param cameras：相机内参集合，包含畸变模型参数（径向畸变、切向畸变等）
 * @param images：图像集合，包含原始特征点坐标（畸变坐标）
 * @param clean_points：是否强制重新去畸变，true=重新处理所有图像，false=跳过已处理的图像
 */
void UndistortImages(std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points) {
  // ========== 步骤1：收集需要去畸变的图像 ==========
  std::vector<image_t> image_ids;
  for (auto& [image_id, image] : images) {
    const int num_points = image.features.size();
    // 如果图像已经去畸变过（features_undist已填充）且不需要清理，则跳过
    if (image.features_undist.size() == num_points && !clean_points)
      continue;  // 已经完成去畸变处理
    image_ids.push_back(image_id);
  }

  // ========== 步骤2：创建线程池进行并行处理 ==========
  // 去畸变计算是独立的，可以并行处理以提高效率
  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);

  LOG(INFO) << "Undistorting images..";
  const int num_images = image_ids.size();
  
  // ========== 步骤3：为每个图像创建去畸变任务 ==========
  for (int image_idx = 0; image_idx < num_images; image_idx++) {
    Image& image = images[image_ids[image_idx]];
    const int num_points = image.features.size();
    // 再次检查是否已经去畸变（双重检查，确保线程安全）
    if (image.features_undist.size() == num_points && !clean_points)
      continue;  // 已经完成去畸变处理

    // 获取该图像对应的相机内参和畸变模型
    const Camera& camera = cameras[image.camera_id];

    // 添加去畸变任务到线程池
    thread_pool.AddTask([&image, &camera, num_points]() {
      // 清空并预分配存储空间
      image.features_undist.clear();
      image.features_undist.reserve(num_points);
      
      // 对每个特征点进行去畸变处理
      for (int i = 0; i < num_points; i++) {
        // ===== 去畸变过程 =====
        // 1. camera.CamFromImg(): 将畸变的像素坐标转换为无畸变的归一化相机坐标
        //    输入：image.features[i] - 2D像素坐标 (u, v)
        //    处理：逆畸变模型 + 去除内参影响
        //    输出：归一化相机坐标 (x, y)，其中 x = (u - cx)/fx, y = (v - cy)/fy（去畸变后）
        // 2. .value_or(Eigen::Vector2d::Zero()): 如果去畸变失败（例如点在畸变模型有效范围外），使用零向量
        // 3. .homogeneous(): 转换为齐次坐标 (x, y) -> (x, y, 1)
        // 4. .normalized(): 归一化为单位向量 (x, y, 1) / ||(x, y, 1)||
        //    这样做是为了统一表示，方便后续的几何计算（如本质矩阵估计）
        image.features_undist.emplace_back(
            camera.CamFromImg(image.features[i])
                .value_or(Eigen::Vector2d::Zero())
                .homogeneous()
                .normalized());
      }
    });
  }

  // ========== 步骤4：等待所有任务完成 ==========
  thread_pool.Wait();
  LOG(INFO) << "Image undistortion done";
}

}  // namespace glomap
