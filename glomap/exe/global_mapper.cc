#include "glomap/controllers/global_mapper.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/pose_io.h"
#include "glomap/types.h"

#include <colmap/util/file.h>
#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace glomap {
// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
/**
 * [功能描述]：运行全局映射器（Global Mapper）的主函数，负责执行完整的3D重建流程
 * @param argc：命令行参数数量
 * @param argv：命令行参数数组
 * @return 返回执行状态：EXIT_SUCCESS 表示成功，EXIT_FAILURE 表示失败
 */
int RunMapper(int argc, char** argv) {
  // ========== 1. 初始化参数变量 ==========
  std::string database_path;    // 输入数据库路径（必需参数）
  std::string output_path;      // 输出重建结果路径（必需参数）

  std::string image_path = "";  // 图像路径（可选参数，用于导出）
  std::string constraint_type = "ONLY_POINTS";  // 约束类型（默认仅使用点约束）
  std::string output_format = "bin";            // 输出格式（默认二进制格式）

  // ========== 2. 配置选项管理器 ==========
  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);  // 添加必需选项：数据库路径
  options.AddRequiredOption("output_path", &output_path);      // 添加必需选项：输出路径
  options.AddDefaultOption("image_path", &image_path);         // 添加可选选项：图像路径
  // 添加约束类型选项，支持四种类型：仅点、仅相机、点和相机平衡、点和相机
  options.AddDefaultOption("constraint_type",
                           &constraint_type,
                           "{ONLY_POINTS, ONLY_CAMERAS, "
                           "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");  // 添加输出格式选项
  options.AddGlobalMapperFullOptions();  // 添加全局映射器的所有配置选项

  // 解析命令行参数
  options.Parse(argc, argv);

  // ========== 3. 验证输入参数 ==========
  // 检查数据库文件是否存在
  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  // 根据字符串设置约束类型枚举值
  if (constraint_type == "ONLY_POINTS") {
    // 仅使用3D点作为约束（最常用）
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    // 仅使用相机位置作为约束
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    // 使用点和相机约束，并进行平衡
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    // 同时使用点和相机约束
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // 检查输出格式是否有效（仅支持二进制或文本格式）
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // ========== 4. 加载数据库 ==========
  ViewGraph view_graph;                             // 视图图：存储图像间的相对位姿关系
  std::unordered_map<rig_t, Rig> rigs;              // 相机组（rig）集合
  std::unordered_map<camera_t, Camera> cameras;     // 相机内参集合
  std::unordered_map<frame_t, Frame> frames;        // 帧数据集合
  std::unordered_map<image_t, Image> images;        // 图像数据集合（包含位姿等信息）
  std::unordered_map<track_t, Track> tracks;        // 3D轨迹（特征点对应的3D点）集合

  // 打开COLMAP数据库
  auto database = colmap::Database::Open(database_path);
  // 将COLMAP数据库格式转换为GLOMAP内部数据结构
  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  // 检查是否有图像对信息（这是重建的基础）
  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  // ========== 5. 创建全局映射器实例 ==========
  GlobalMapper global_mapper(*options.mapper);

  // ========== 6. 执行主求解过程 ==========
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;  // 创建计时器，用于统计重建耗时
  run_timer.Start();        // 开始计时
  // 核心函数：执行全局SfM重建，包括旋转平均、位置估计、三角化、BA优化等步骤
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);
  run_timer.Pause();  // 暂停计时

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  // ========== 7. 导出重建结果 ==========
  // 将重建结果写入COLMAP格式的输出文件
  WriteGlomapReconstruction(output_path,
                            rigs,
                            cameras,
                            frames,
                            images,
                            tracks,
                            output_format,
                            image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Mappers starting from COLMAP reconstruction
// -------------------------------------
int RunMapperResume(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperResumeFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the reconstruction
  ViewGraph view_graph;                        // dummy variable
  std::shared_ptr<colmap::Database> database;  // dummy variable

  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, rigs, cameras, frames, images, tracks);

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(output_path,
                            rigs,
                            cameras,
                            frames,
                            images,
                            tracks,
                            output_format,
                            image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap
