#include "glomap/io/colmap_io.h"

#include <colmap/util/file.h>
#include <colmap/util/misc.h>

namespace glomap {

/**
 * [功能描述]：将Glomap重建结果写入COLMAP格式文件
 * 该函数将Glomap的重建数据（相机、帧、图像、轨迹等）转换为COLMAP格式并导出。
 * 如果存在多个聚类簇（clusters），则分别导出每个簇的重建结果。
 * 
 * @param reconstruction_path：重建结果的输出路径
 * @param rigs：相机装备映射表，键为装备ID，值为Rig对象
 * @param cameras：相机参数映射表，键为相机ID，值为Camera对象
 * @param frames：帧数据映射表，键为帧ID，值为Frame对象
 * @param images：图像数据映射表，键为图像ID，值为Image对象
 * @param tracks：3D点轨迹映射表，键为轨迹ID，值为Track对象
 * @param output_format：输出格式，支持"txt"（文本格式）或"bin"（二进制格式）
 * @param image_path：图像文件路径，用于提取点云颜色信息。如果为空则不提取颜色
 * @return 无返回值
 */
void WriteGlomapReconstruction(
    const std::string& reconstruction_path,
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    const std::string output_format,
    const std::string image_path) {
  // 检查是否应用了重建剪枝（reconstruction pruning）
  // 如果有多个聚类簇，则需要分别导出每个簇的重建结果
  int largest_component_num = -1;
  
  // 遍历所有帧，查找最大的聚类簇ID
  for (const auto& [frame_id, frame] : frames) {
    if (frame.cluster_id > largest_component_num)
      largest_component_num = frame.cluster_id;
  }
  
  // 如果没有分成多个聚类簇（largest_component_num == -1），则整体输出
  if (largest_component_num == -1) {
    // 创建COLMAP重建对象
    colmap::Reconstruction reconstruction;
    
    // 将Glomap格式数据转换为COLMAP格式
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    
    // 如果提供了图像路径，则从图像中提取点云颜色信息
    if (image_path != "") {
      LOG(INFO) << "Extracting colors ...";
      reconstruction.ExtractColorsForAllImages(image_path);
    }
    
    // 创建输出目录，路径为 reconstruction_path/0
    colmap::CreateDirIfNotExists(reconstruction_path + "/0", true);
    
    // 根据指定的输出格式保存重建结果
    if (output_format == "txt") {
      // 以文本格式输出
      reconstruction.WriteText(reconstruction_path + "/0");
    } else if (output_format == "bin") {
      // 以二进制格式输出
      reconstruction.WriteBinary(reconstruction_path + "/0");
    } else {
      LOG(ERROR) << "Unsupported output type";
    }
  } else {
    // 存在多个聚类簇，分别导出每个簇的重建结果
    // 遍历从0到largest_component_num的所有聚类簇
    for (int comp = 0; comp <= largest_component_num; comp++) {
      // 显示导出进度（使用\r实现同行刷新）
      std::cout << "\r Exporting reconstruction " << comp + 1 << " / "
                << largest_component_num + 1 << std::flush;
      
      // 为当前聚类簇创建COLMAP重建对象
      colmap::Reconstruction reconstruction;
      
      // 转换当前聚类簇的数据，最后一个参数comp指定要导出的聚类簇ID
      ConvertGlomapToColmap(
          rigs, cameras, frames, images, tracks, reconstruction, comp);
      
      // 如果提供了图像路径，则提取当前簇的点云颜色
      if (image_path != "") {
        reconstruction.ExtractColorsForAllImages(image_path);
      }
      
      // 为当前聚类簇创建单独的输出目录，路径为 reconstruction_path/comp
      colmap::CreateDirIfNotExists(
          reconstruction_path + "/" + std::to_string(comp), true);
      
      // 根据指定的输出格式保存当前簇的重建结果
      if (output_format == "txt") {
        // 以文本格式输出到对应的子目录
        reconstruction.WriteText(reconstruction_path + "/" +
                                 std::to_string(comp));
      } else if (output_format == "bin") {
        // 以二进制格式输出到对应的子目录
        reconstruction.WriteBinary(reconstruction_path + "/" +
                                   std::to_string(comp));
      } else {
        LOG(ERROR) << "Unsupported output type";
      }
    }
    // 输出换行符，结束进度显示
    std::cout << std::endl;
  }
}

void WriteColmapReconstruction(const std::string& reconstruction_path,
                               const colmap::Reconstruction& reconstruction,
                               const std::string output_format) {
  colmap::CreateDirIfNotExists(reconstruction_path, true);
  if (output_format == "txt") {
    reconstruction.WriteText(reconstruction_path);
  } else if (output_format == "bin") {
    reconstruction.WriteBinary(reconstruction_path);
  } else {
    LOG(ERROR) << "Unsupported output type";
  }
}

}  // namespace glomap
