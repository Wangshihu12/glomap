#include "global_mapper.h"

#include "glomap/controllers/rotation_averager.h"
#include "glomap/io/colmap_converter.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/image_undistorter.h"
#include "glomap/processors/reconstruction_normalizer.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/relpose_filter.h"
#include "glomap/processors/track_filter.h"
#include "glomap/processors/view_graph_manipulation.h"

#include <colmap/util/file.h>
#include <colmap/util/timer.h>

namespace glomap {

// TODO: Rig normalizaiton has not be done
/**
 * [功能描述]：全局SfM重建的核心求解函数，执行完整的3D重建流程
 * @param database：COLMAP数据库的引用，包含特征匹配等信息
 * @param view_graph：视图图，存储图像间的连接关系和相对位姿
 * @param rigs：相机组（rig）的集合，用于多相机系统
 * @param cameras：相机内参的集合，包含焦距、畸变等参数
 * @param frames：帧数据的集合
 * @param images：图像数据的集合，包含图像的位姿、特征点等信息
 * @param tracks：3D轨迹的集合，每个轨迹对应一个3D点及其在多幅图像中的观测
 * @return 返回是否成功完成重建：true表示成功，false表示失败
 */
bool GlobalMapper::Solve(const colmap::Database& database,
                         ViewGraph& view_graph,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<frame_t, Frame>& frames,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks) {
  // ========== 步骤0：预处理 ==========
  // 对输入数据进行初步处理，更新图像对配置并分解相对位姿
  if (!options_.skip_preprocessing) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running preprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // 如果相机内参质量较好，强制使用本质矩阵（Essential Matrix）而非基础矩阵
    // 这可以提高相对位姿估计的精度
    ViewGraphManipulater::UpdateImagePairsConfig(view_graph, cameras, images);
    // 将相对位姿分解为旋转和平移分量
    ViewGraphManipulater::DecomposeRelPose(view_graph, cameras, images);
    run_timer.PrintSeconds();
  }

  // ========== 步骤1：视图图校准 ==========
  // 优化相机内参（如焦距）以提高后续步骤的精度
  if (!options_.skip_view_graph_calibration) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running view graph calibration ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    // 创建视图图校准引擎，用于全局优化相机参数
    ViewGraphCalibrator vgcalib_engine(options_.opt_vgcalib);
    // 执行校准求解，如果失败则终止重建
    if (!vgcalib_engine.Solve(view_graph, cameras, images)) {
      return false;
    }
  }

  // ========== 步骤2：相对位姿估计 ==========
  // 估计图像对之间的相对旋转和平移，构建视图图的边
  // TODO: 为相机组（rigs）使用广义相对位姿估计
  if (!options_.skip_relative_pose_estimation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running relative pose estimation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // 相对位姿估计依赖于去畸变后的图像，因此先进行去畸变处理
    UndistortImages(cameras, images, true);
    // 估计所有图像对的相对位姿（旋转和平移）
    EstimateRelativePoses(view_graph, cameras, images, options_.opt_relpose);

    InlierThresholdOptions inlier_thresholds = options_.inlier_thresholds;
    // 统计每个图像对的内点数量，用于后续过滤
    ImagePairsInlierCount(view_graph, cameras, images, inlier_thresholds, true);

    // 根据内点数量过滤图像对，移除内点数量过少的边
    RelPoseFilter::FilterInlierNum(view_graph,
                                   options_.inlier_thresholds.min_inlier_num);
    // 根据内点比例过滤图像对，移除内点比例过低的边
    RelPoseFilter::FilterInlierRatio(
        view_graph, options_.inlier_thresholds.min_inlier_ratio);

    // 保留最大连通分量，移除孤立的图像
    if (view_graph.KeepLargestConnectedComponents(frames, images) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    run_timer.PrintSeconds();
  }

  // ========== 步骤3：旋转平均 ==========
  // 从相对旋转估计全局旋转，这是全局SfM的核心步骤之一
  // 采用两次迭代策略：第一次用于过滤异常值，第二次用于最终估计
  if (!options_.skip_rotation_averaging) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running rotation averaging ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    // 第一次运行：用于过滤不一致的相对旋转
    SolveRotationAveraging(view_graph, rigs, frames, images, options_.opt_ra);

    // 根据旋转误差过滤图像对，移除与全局旋转不一致的边
    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    // 重新检查连通性，确保仍有连通分量
    if (view_graph.KeepLargestConnectedComponents(frames, images) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    // 第二次运行：在过滤后的数据上进行最终的旋转估计
    if (!SolveRotationAveraging(
            view_graph, rigs, frames, images, options_.opt_ra)) {
      return false;
    }
    // 再次过滤旋转误差较大的边
    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    // 保留最大连通分量，并统计最终的图像数量
    image_t num_img = view_graph.KeepLargestConnectedComponents(frames, images);
    if (num_img == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }
    LOG(INFO) << num_img << " / " << images.size()
              << " images are within the connected component." << std::endl;

    run_timer.PrintSeconds();
  }

  // ========== 步骤4：轨迹建立和选择 ==========
  // 将多幅图像中的特征点关联起来，形成3D点的观测轨迹
  if (!options_.skip_track_establishment) {
    colmap::Timer run_timer;
    run_timer.Start();

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running track establishment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    // 创建轨迹引擎，用于建立特征点的对应关系
    TrackEngine track_engine(view_graph, images, options_.opt_track);
    std::unordered_map<track_t, Track> tracks_full;  // 存储所有轨迹（包括低质量的）
    // 建立完整的特征点轨迹，将不同图像中对应的特征点链接起来
    track_engine.EstablishFullTracks(tracks_full);

    // 过滤轨迹，选择高质量的轨迹用于后续优化
    // 低质量的轨迹（如观测数太少、重投影误差大等）会被剔除
    track_t num_tracks = track_engine.FindTracksForProblem(tracks_full, tracks);
    LOG(INFO) << "Before filtering: " << tracks_full.size()
              << ", after filtering: " << num_tracks << std::endl;

    run_timer.PrintSeconds();
  }

  // ========== 步骤5：全局定位 ==========
  // 估计所有相机的全局位置（平移），此时旋转已知
  if (!options_.skip_global_positioning) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running global positioning ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // 检查约束类型，当前仅支持使用3D点作为约束
    if (options_.opt_gp.constraint_type !=
        GlobalPositionerOptions::ConstraintType::ONLY_POINTS) {
      LOG(ERROR) << "Only points are used for solving camera positions";
      return false;
    }

    colmap::Timer run_timer;
    run_timer.Start();
    // 如果前面步骤被跳过，需要在此处进行图像去畸变
    // 如果已经去畸变过的图像会被跳过（false参数）
    UndistortImages(cameras, images, false);

    // 创建全局定位求解器
    GlobalPositioner gp_engine(options_.opt_gp);

    // 求解相机的全局位置（使用3D点和已知的旋转作为约束）
    // TODO: 考虑支持其他模式（如使用相机约束）
    if (!gp_engine.Solve(view_graph, rigs, cameras, frames, images, tracks)) {
      return false;
    }
    // 根据角度误差过滤轨迹，移除与估计位姿不一致的观测
    TrackFilter::FilterTracksByAngle(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_angle_error);

    // 根据三角化角度过滤轨迹，移除三角化角度过小的点（几何约束弱）
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
    // 根据重投影误差过滤轨迹，设置较大的阈值以避免过度过滤
    // 使用10倍的阈值，因为此时是初步估计，后续BA会进一步优化
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        10 * options_.inlier_thresholds.max_reprojection_error);
    // 归一化重建结果（统一尺度和坐标系）
    // 注意：如果使用相机组（rig），则不需要归一化
    NormalizeReconstruction(rigs, cameras, frames, images, tracks);

    run_timer.PrintSeconds();
  }

  // ========== 步骤6：光束法平差（Bundle Adjustment, BA） ==========
  // 联合优化相机位姿和3D点位置，最小化重投影误差
  if (!options_.skip_bundle_adjustment) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running bundle adjustment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    LOG(INFO) << "Bundle adjustment start" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    // 迭代执行多轮BA，每轮包括优化和过滤两个步骤
    for (int ite = 0; ite < options_.num_iteration_bundle_adjustment; ite++) {
      BundleAdjuster ba_engine(options_.opt_ba);

      BundleAdjusterOptions& ba_engine_options_inner = ba_engine.GetOptions();

      // ===== 分阶段的光束法平差 =====
      // 6.1. 第一阶段：仅优化相机位置（平移），保持旋转固定
      // 这种策略可以提高稳定性，避免陷入局部最优
      ba_engine_options_inner.optimize_rotations = false;
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 1 finished (position only)";
      run_timer.PrintSeconds();

      // 6.2. 第二阶段：如果需要，同时优化旋转和位置
      // 在第一阶段稳定后，再优化旋转可以获得更好的结果
      ba_engine_options_inner.optimize_rotations =
          options_.opt_ba.optimize_rotations;
      if (ba_engine_options_inner.optimize_rotations &&
          !ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 2 finished";
      if (ite != options_.num_iteration_bundle_adjustment - 1)
        run_timer.PrintSeconds();

      // 再次归一化重建结构
      NormalizeReconstruction(rigs, cameras, frames, images, tracks);

      // 6.3. 根据优化结果过滤轨迹
      // 采用渐进式过滤策略：每轮迭代中，异常值判定标准逐渐收紧
      // 如果过滤掉的轨迹很少，则使用更严格的标准继续过滤，而不是立即开始新一轮BA
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";

      bool status = true;  // 标记是否需要继续BA迭代
      size_t filtered_num = 0;  // 累计过滤掉的轨迹数量
      // 自适应过滤循环：逐步收紧阈值
      while (status && ite < options_.num_iteration_bundle_adjustment) {
        // 计算缩放因子：第0轮用3倍阈值，第1轮用2倍，第2轮及以后用1倍
        double scaling = std::max(3 - ite, 1);
        // 根据重投影误差过滤轨迹，阈值随迭代次数递减
        filtered_num += TrackFilter::FilterTracksByReprojection(
            view_graph,
            cameras,
            images,
            tracks,
            scaling * options_.inlier_thresholds.max_reprojection_error);

        // 如果过滤掉的轨迹超过总数的0.1%，说明还有较多异常值，需要重新BA
        if (filtered_num > 1e-3 * tracks.size()) {
          status = false;
        } else
          ite++;  // 否则跳过当前迭代，使用更严格的阈值继续过滤
      }
      // 如果过滤掉的轨迹极少（<0.1%），提前终止BA迭代
      if (status) {
        LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
        break;
      }
    }

    // BA迭代完成后，进行最终的轨迹过滤
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    // 使用最严格的阈值过滤重投影误差较大的轨迹
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    // 过滤三角化角度过小的轨迹（几何约束弱，不可靠）
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);

    run_timer.PrintSeconds();
  }

  // ========== 步骤7：重三角化 ==========
  // 尝试为更多的特征点轨迹生成3D点，增加重建的完整性
  if (!options_.skip_retriangulation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running retriangulation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    // 执行多轮重三角化，每轮包括三角化、BA优化、过滤三个步骤
    for (int ite = 0; ite < options_.num_iteration_retriangulation; ite++) {
      colmap::Timer run_timer;
      run_timer.Start();
      // 对之前未能成功三角化的轨迹重新进行三角化
      // 随着相机位姿的优化，一些轨迹可能变得可以三角化
      RetriangulateTracks(options_.opt_triangulator,
                          database,
                          rigs,
                          cameras,
                          frames,
                          images,
                          tracks);
      run_timer.PrintSeconds();

      std::cout << "-------------------------------------" << std::endl;
      std::cout << "Running bundle adjustment ..." << std::endl;
      std::cout << "-------------------------------------" << std::endl;
      LOG(INFO) << "Bundle adjustment start" << std::endl;
      // 对新增的3D点和相机位姿进行BA优化
      BundleAdjuster ba_engine(options_.opt_ba);
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }

      // 根据优化结果过滤轨迹
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";
      // 移除重投影误差较大的轨迹
      TrackFilter::FilterTracksByReprojection(
          view_graph,
          cameras,
          images,
          tracks,
          options_.inlier_thresholds.max_reprojection_error);
      // 再次运行BA以稳定结果
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }
      run_timer.PrintSeconds();
    }

    // 归一化重建结构
    NormalizeReconstruction(rigs, cameras, frames, images, tracks);

    // 重三角化完成后，进行最终的轨迹过滤
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    // 根据重投影误差过滤
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    // 根据三角化角度过滤，确保几何约束强
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
  }

  // ========== 步骤8：重建剪枝（后处理） ==========
  // 移除质量较差的图像和轨迹，提高重建结果的可靠性
  if (!options_.skip_pruning) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running postprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    // 剪枝弱连接的图像
    // 移除与其他图像连接较少（观测到的共同3D点少）的图像
    // 这些图像通常位姿估计不准确，对重建贡献小
    PruneWeaklyConnectedImages(frames, images, tracks);

    run_timer.PrintSeconds();
  }

  // 所有步骤成功完成，返回true
  return true;
}

}  // namespace glomap
