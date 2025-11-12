#include "glomap/controllers/rotation_averager.h"

#include "glomap/estimators/rotation_initializer.h"
#include "glomap/io/colmap_converter.h"

namespace glomap {

/**
 * [功能描述]：全局旋转平均的求解函数，从相对旋转估计全局绝对旋转
 * 这是全局SfM的核心步骤之一，支持重力先验和相机组（rig）等高级特性
 * @param view_graph：视图图，包含图像对及其相对旋转
 * @param rigs：相机组集合，用于多相机系统
 * @param frames：帧数据集合
 * @param images：图像数据集合，存储估计的全局旋转
 * @param options：旋转平均的配置选项（重力、分层策略等）
 * @return 返回是否成功求解：true表示成功，false表示失败
 */
bool SolveRotationAveraging(ViewGraph& view_graph,
                            std::unordered_map<rig_t, Rig>& rigs,
                            std::unordered_map<frame_t, Frame>& frames,
                            std::unordered_map<image_t, Image>& images,
                            const RotationAveragerOptions& options) {
  // ========== 步骤1：保留最大连通分量 ==========
  // 确保所有图像都通过有效的相对旋转连接在一起
  view_graph.KeepLargestConnectedComponents(frames, images);

  // ========== 步骤2：判断是否使用1自由度优化（利用重力先验） ==========
  // 当有重力信息时，可以约束Z轴方向，将3自由度旋转问题简化为1自由度（仅绕重力轴旋转）
  bool solve_1dof_system = options.use_gravity && options.use_stratified;

  ViewGraph view_graph_grav;  // 仅包含带重力信息的图像对
  image_pair_t total_pairs = 0;
  if (solve_1dof_system) {
    // 准备两个子集：
    // 1. 带重力信息的图像对 -> 用于1自由度优化
    // 2. 不带重力信息的图像对 -> 用于3自由度优化
    // 分层策略：先解决带重力约束的子问题，再在统一系统中求解全部
    for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (!image_pair.is_valid) continue;

      const Image& image1 = images[image_pair.image_id1];
      const Image& image2 = images[image_pair.image_id2];

      if (!image1.IsRegistered() || !image2.IsRegistered()) continue;

      total_pairs++;

      // 如果两个图像都有重力信息，加入重力子集
      // 重力信息通常来自IMU传感器或EXIF数据
      if (image1.HasGravity() && image2.HasGravity()) {
        view_graph_grav.image_pairs.emplace(
            pair_id,
            ImagePair(image_pair.image_id1,
                      image_pair.image_id2,
                      image_pair.cam2_from_cam1));
      }
    }
  }

  const size_t grav_pairs = view_graph_grav.image_pairs.size();

  LOG(INFO) << "Total image pairs: " << total_pairs
            << ", gravity image pairs: " << grav_pairs;

  // 决定是否真正使用1自由度系统
  // 如果没有重力图像对，或者95%以上的图像对都有重力，则直接使用标准3自由度方法
  // 原因：全部或几乎全部有重力时，分层策略不会带来额外好处
  const bool status = grav_pairs == 0 || grav_pairs > total_pairs * 0.95;
  solve_1dof_system = solve_1dof_system && !status;

  // ========== 步骤3：运行1自由度旋转优化（如果适用） ==========
  if (solve_1dof_system) {
    // 在混合先验系统中，先解决1自由度的子问题
    // 1自由度：只优化绕重力轴的旋转角度，Z轴方向由重力约束
    LOG(INFO) << "Solving subset 1DoF rotation averaging problem in the mixed "
                 "prior system";
    // 保持重力视图图的连通性
    view_graph_grav.KeepLargestConnectedComponents(frames, images);
    RotationEstimator rotation_estimator_grav(options);
    // 估计带重力约束的图像的旋转
    // 这些旋转将作为后续全局优化的初始值
    if (!rotation_estimator_grav.EstimateRotations(
            view_graph_grav, rigs, frames, images)) {
      return false;
    }
    // 重新检查完整视图图的连通性
    view_graph.KeepLargestConnectedComponents(frames, images);
  }

  // ========== 步骤4：识别相机组中未知变换的相机 ==========
  // 对于相机组（rig）系统，需要处理cam_from_rig变换未知的相机
  // cam_from_rig：相机相对于rig参考系的固定变换（外参）
  std::unordered_set<camera_t> unknown_cams_from_rig;
  rig_t max_rig_id = 0;
  for (const auto& [rig_id, rig] : rigs) {
    max_rig_id = std::max(max_rig_id, rig_id);
    // 遍历相机组中的非参考传感器（参考传感器的变换定义为单位阵）
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      // 如果cam_from_rig变换未知，需要特殊处理
      // 这种情况在多相机系统标定不完整时会出现
      if (!rig.MaybeSensorFromRig(sensor_id).has_value()) {
        unknown_cams_from_rig.insert(sensor_id.id);
      }
    }
  }

  bool status_ra = false;
  // ========== 步骤5：运行平凡旋转平均（处理未知cam_from_rig的相机） ==========
  if (!unknown_cams_from_rig.empty() && !options.skip_initialization) {
    LOG(INFO) << "Running trivial rotation averaging for rigged cameras";
    // 创建简化的rig系统用于初始化
    // 策略：将每个未知相机视为独立的rig，先估计其旋转，再转换回原rig系统
    std::unordered_map<rig_t, Rig> rigs_trivial;
    std::unordered_map<frame_t, Frame> frames_trivial;
    std::unordered_map<image_t, Image> images_trivial;

    // ===== 子步骤5.1：为已知cam_from_rig的相机创建简化rig =====
    std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
    for (const auto& [rig_id, rig] : rigs) {
      Rig rig_trivial;
      rig_trivial.SetRigId(rig_id);
      rig_trivial.AddRefSensor(rig.RefSensorId());
      camera_id_to_rig_id[rig.RefSensorId().id] = rig_id;

      // 只添加已知cam_from_rig变换的传感器
      for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
        if (sensor_id.type != SensorType::CAMERA) continue;
        if (rig.MaybeSensorFromRig(sensor_id).has_value()) {
          rig_trivial.AddSensor(sensor_id, sensor);
          camera_id_to_rig_id[sensor_id.id] = rig_id;
        }
      }
      rigs_trivial[rig_trivial.RigId()] = rig_trivial;
    }

    // ===== 子步骤5.2：为每个未知cam_from_rig的相机创建独立的平凡rig =====
    // 平凡rig：只包含一个相机，cam_from_rig为单位阵
    // 这样可以直接估计该相机的绝对旋转，无需已知的rig变换
    for (const auto& camera_id : unknown_cams_from_rig) {
      Rig rig_trivial;
      rig_trivial.SetRigId(++max_rig_id);
      rig_trivial.AddRefSensor(sensor_t(SensorType::CAMERA, camera_id));
      rigs_trivial[rig_trivial.RigId()] = rig_trivial;
      camera_id_to_rig_id[camera_id] = rig_trivial.RigId();
    }

    // ===== 子步骤5.3：创建平凡帧结构 =====
    frame_t max_frame_id = 0;
    for (const auto& [frame_id, _] : frames) {
      THROW_CHECK_NE(frame_id, colmap::kInvalidFrameId);
      max_frame_id = std::max(max_frame_id, frame_id);
    }
    max_frame_id++;

    // 为每个帧创建对应的平凡帧
    for (auto& [frame_id, frame] : frames) {
      Frame frame_trivial = Frame();
      frame_trivial.SetFrameId(frame_id);
      frame_trivial.SetRigId(frame.RigId());
      frame_trivial.SetRigPtr(rigs_trivial.find(frame.RigId()) !=
                                      rigs_trivial.end()
                                  ? &rigs_trivial[frame.RigId()]
                                  : nullptr);
      frames_trivial[frame_id] = frame_trivial;

      // 处理帧中的每个图像
      for (const auto& data_id : frame.ImageIds()) {
        const auto& image = images.at(data_id.id);
        if (!image.IsRegistered()) continue;
        auto& image_trivial =
            images_trivial
                .emplace(data_id.id,
                         Image(data_id.id, image.camera_id, image.file_name))
                .first->second;

        // 根据相机是否在未知集合中采取不同策略
        if (unknown_cams_from_rig.find(image_trivial.camera_id) ==
            unknown_cams_from_rig.end()) {
          // 已知cam_from_rig的相机：保持在原帧中
          frames_trivial[frame_id].AddDataId(image_trivial.DataId());
          image_trivial.frame_id = frame_id;
          image_trivial.frame_ptr = &frames_trivial[frame_id];
        } else {
          // 未知cam_from_rig的相机：创建独立的平凡帧
          // 这样每个未知相机都有自己的帧，可以独立估计旋转
          CreateFrameForImage(Rigid3d(),
                              image_trivial,
                              rigs_trivial,
                              frames_trivial,
                              camera_id_to_rig_id[image.camera_id],
                              max_frame_id);
          max_frame_id++;
        }
      }
    }

    view_graph.KeepLargestConnectedComponents(frames_trivial, images_trivial);
    
    // ===== 子步骤5.4：运行平凡旋转平均 =====
    RotationEstimatorOptions options_trivial = options;
    options_trivial.skip_initialization = options.skip_initialization;
    RotationEstimator rotation_estimator_trivial(options_trivial);
    // 在平凡系统中估计旋转（每个未知相机独立优化）
    rotation_estimator_trivial.EstimateRotations(
        view_graph, rigs_trivial, frames_trivial, images_trivial);

    // ===== 子步骤5.5：收集平凡系统的结果 =====
    std::unordered_map<image_t, Rigid3d> cams_from_world;
    for (const auto& [image_id, image] : images_trivial) {
      if (!image.IsRegistered()) continue;
      cams_from_world[image_id] = image.CamFromWorld();
    }

    // ===== 子步骤5.6：将图像旋转转换回原始rig系统 =====
    // 从独立估计的图像旋转，推导出rig的旋转和未知的cam_from_rig变换
    ConvertRotationsFromImageToRig(cams_from_world, images, rigs, frames);

    // ===== 子步骤5.7：在原始rig系统中精化旋转 =====
    RotationEstimatorOptions options_ra = options;
    options_ra.skip_initialization = true;  // 使用平凡系统的结果作为初始值
    RotationEstimator rotation_estimator(options_ra);
    status_ra =
        rotation_estimator.EstimateRotations(view_graph, rigs, frames, images);
    view_graph.KeepLargestConnectedComponents(frames, images);
  } else {
    // ========== 步骤6：运行标准旋转平均 ==========
    // 当没有未知cam_from_rig的相机，或者跳过初始化时，直接运行标准方法
    RotationAveragerOptions options_ra = options;
    // 如果存在未知cam_from_rig的相机但没有运行平凡初始化，
    // 需要禁用skip_initialization以确保收敛
    if (unknown_cams_from_rig.size() > 0) {
      options_ra.skip_initialization = false;
    }

    RotationEstimator rotation_estimator(options_ra);
    status_ra =
        rotation_estimator.EstimateRotations(view_graph, rigs, frames, images);
    view_graph.KeepLargestConnectedComponents(frames, images);
  }
  return status_ra;
}

}  // namespace glomap
