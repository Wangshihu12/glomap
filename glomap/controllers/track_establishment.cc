#include "track_establishment.h"

namespace glomap {

/**
 * [功能描述]：建立完整的特征点轨迹，将不同图像中对应的特征点连接起来
 * 轨迹（Track）：在多幅图像中观测到的同一个3D点的所有2D特征点的集合
 * 例如：同一个场景点在图像1、3、5中被观测到，这三个特征点构成一条轨迹
 * @param tracks：输出参数，存储所有建立的轨迹，每条轨迹包含多个图像的观测
 * @return 返回建立的轨迹数量
 */
size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<track_t, Track>& tracks) {
  // 清空输出容器和并查集结构
  tracks.clear();
  uf_.Clear();  // uf_: Union-Find并查集，用于高效连接特征点

  // ========== 步骤1：盲目连接 ==========
  // 使用并查集将所有匹配的特征点连接起来
  // "盲目"是指不考虑几何一致性，只要有匹配就连接
  // 这一步会将跨多幅图像的对应特征点合并到同一个集合中
  BlindConcatenation();

  // ========== 步骤2：轨迹收集 ==========
  // 遍历并查集的结果，将每个连通分量转换为一条轨迹
  // 每个连通分量代表一个3D点在多幅图像中的所有观测
  TrackCollection(tracks);

  return tracks.size();
}

/**
 * [功能描述]：盲目连接所有匹配的特征点，使用并查集建立初始轨迹
 * "盲目"是指仅依据内点标记连接特征点，不进行额外的几何一致性检查
 * 算法：遍历所有图像对的内点匹配，将对应的特征点在并查集中合并
 */
void TrackEngine::BlindConcatenation() {
  // 初始化并查集数据结构，连接所有对应关系
  image_pair_t counter = 0;
  // 遍历视图图中的所有图像对
  for (auto pair : view_graph_.image_pairs) {
    // 显示进度信息（每1000个图像对或最后一个）
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Initializing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const ImagePair& image_pair = pair.second;
    // 跳过无效的图像对（如几何验证失败的）
    if (!image_pair.is_valid) continue;

    // 获取特征匹配矩阵
    // matches: N x 2矩阵，每行[idx1, idx2]表示图像1的特征idx1与图像2的特征idx2匹配
    const Eigen::MatrixXi& matches = image_pair.matches;

    // 获取内点掩码
    // inliers: 索引数组，存储满足几何约束的匹配的索引
    // 只有内点才会被用于建立轨迹，外点（误匹配）会被忽略
    const std::vector<int>& inliers = image_pair.inliers;

    // 遍历所有内点匹配
    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];  // 获取匹配在matches矩阵中的索引

      // 获取匹配的特征点在各自图像中的局部索引
      const uint32_t& point1_idx = matches(idx, 0);  // 图像1中的特征点索引
      const uint32_t& point2_idx = matches(idx, 1);  // 图像2中的特征点索引

      // ===== 计算全局唯一ID =====
      // 为每个特征点分配一个全局唯一的64位ID
      // 高32位：图像ID，低32位：特征点在该图像中的索引
      // 这样可以唯一标识整个场景中的任意一个特征点观测
      // 例如：图像5的第100个特征点 -> (5 << 32) | 100
      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      // ===== 使用并查集连接对应的特征点 =====
      // Union操作将两个特征点标记为属于同一条轨迹
      // 策略：总是将较小的ID作为root（保持并查集结构的一致性）
      if (point_global_id2 < point_global_id1) {
        uf_.Union(point_global_id1, point_global_id2);
      } else
        uf_.Union(point_global_id2, point_global_id1);
    }
  }
  std::cout << std::endl;
}

/**
 * [功能描述]：从并查集的连通分量中收集和构建轨迹对象
 * 遍历并查集的结果，将每个连通分量转换为一条Track，并进行一致性检查
 * @param tracks：输出参数，存储收集到的所有轨迹
 */
void TrackEngine::TrackCollection(std::unordered_map<track_t, Track>& tracks) {
  // track_map: 轨迹ID -> 该轨迹包含的所有特征点全局ID
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;
  // track_counter: 轨迹ID -> 该轨迹的匹配数量（用于统计）
  std::unordered_map<uint64_t, int> track_counter;

  // ========== 阶段1：从并查集收集轨迹 ==========
  // 遍历所有图像对，将属于同一连通分量的特征点收集到一起
  size_t counter = 0;
  for (auto pair : view_graph_.image_pairs) {
    // 显示进度信息
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Establishing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) continue;

    // 获取特征匹配矩阵
    const Eigen::MatrixXi& matches = image_pair.matches;

    // 获取内点掩码
    const std::vector<int>& inliers = image_pair.inliers;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // 获取匹配的特征点索引
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      // 计算全局唯一ID（高32位=图像ID，低32位=特征点索引）
      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      // 使用并查集的Find操作找到该特征点所属轨迹的ID（root）
      // 同一轨迹中的所有特征点会有相同的root
      image_pair_t track_id = uf_.Find(point_global_id1);

      // 将两个匹配的特征点都加入该轨迹
      track_map[track_id].insert(point_global_id1);
      track_map[track_id].insert(point_global_id2);
      track_counter[track_id]++;
    }
  }
  std::cout << std::endl;

  // ========== 阶段2：构建Track对象并进行一致性检查 ==========
  counter = 0;
  size_t discarded_counter = 0;  // 统计因不一致而丢弃的轨迹数
  for (auto& [track_id, correspondence_set] : track_map) {
    // 显示进度信息
    if ((counter + 1) % 1000 == 0 || counter == track_map.size() - 1) {
      std::cout << "\r Establishing tracks " << counter + 1 << " / "
                << track_map.size() << std::flush;
    }
    counter++;

    // image_id_set: 图像ID -> 该图像中已添加到当前轨迹的特征点坐标列表
    // 用于检测同一图像中是否有多个特征点被错误地归入同一轨迹
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    
    // 遍历轨迹中的所有特征点
    for (auto point_global_id : correspondence_set) {
      // 解码全局ID：高32位是图像ID，低32位是特征点索引
      image_t image_id = point_global_id >> 32;
      feature_t feature_id = point_global_id & 0xFFFFFFFF;
      
      // ===== 一致性检查 =====
      // 检查当前图像是否已有其他特征点在这条轨迹中
      if (image_id_set.find(image_id) != image_id_set.end()) {
        // 同一图像中有多个特征点在同一轨迹中，这通常是错误的
        // 理论上，一个3D点在同一图像中只能有一个观测
        // 检查这些特征点之间的距离
        for (const auto& feature : image_id_set.at(image_id)) {
          // 如果两个特征点距离超过阈值，说明它们不应该属于同一个3D点
          // 这是由于误匹配或并查集错误合并导致的
          if ((feature - images_.at(image_id).features[feature_id]).norm() >
              options_.thres_inconsistency) {
            // 检测到不一致，清空该轨迹的观测（标记为无效）
            tracks[track_id].observations.clear();
            break;
          }
        }
        // 如果轨迹被标记为无效，丢弃并跳到下一条轨迹
        if (tracks[track_id].observations.size() == 0) {
          discarded_counter++;
          break;
        }
      } else {
        // 这是该图像在此轨迹中的第一个特征点，初始化列表
        image_id_set.insert(
            std::make_pair(image_id, std::vector<Eigen::Vector2d>()));
      }

      // 将特征点坐标添加到检查列表
      image_id_set[image_id].push_back(
          images_.at(image_id).features[feature_id]);

      // 将观测添加到轨迹中（图像ID + 特征点索引）
      tracks[track_id].observations.emplace_back(image_id, feature_id);
    }
  }

  std::cout << std::endl;
  LOG(INFO) << "Discarded " << discarded_counter
            << " tracks due to inconsistency";
}

/**
 * [功能描述]：从完整轨迹集合中选择高质量的轨迹子集用于优化
 * 使用贪心策略确保每个相机都有足够的轨迹观测，同时优先选择观测数多的长轨迹
 * @param tracks_full：完整的轨迹集合（包含所有建立的轨迹）
 * @param tracks_selected：输出参数，选择的高质量轨迹子集
 * @return 返回选择的轨迹数量
 */
size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<track_t, Track>& tracks_full,
    std::unordered_map<track_t, Track>& tracks_selected) {
  // ========== 步骤1：按轨迹长度排序 ==========
  // 轨迹长度 = 观测数量（在多少幅图像中观测到）
  std::vector<std::pair<size_t, track_t>> track_lengths;

  // 收集所有符合条件的轨迹及其长度
  for (const auto& [track_id, track] : tracks_full) {
    // 过滤条件1：观测数必须 >= 最小值（通常为2或3）
    // 观测数太少的轨迹三角化不稳定
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    
    // 过滤条件2：观测数必须 <= 最大值
    // 观测数过多的轨迹可能是误匹配导致的（多条轨迹被错误合并）
    // TODO: 未来需要更优雅的过滤方式
    if (track.observations.size() > options_.max_num_view_per_track) continue;
    
    track_lengths.emplace_back(
        std::make_pair(track.observations.size(), track_id));
  }
  // 按观测数降序排序（长轨迹优先）
  // 长轨迹通常更可靠，因为它们在更多图像中被观测到
  std::sort(std::rbegin(track_lengths), std::rend(track_lengths));

  // ========== 步骤2：初始化每个相机的轨迹计数器 ==========
  std::unordered_map<image_t, track_t> tracks_per_camera;

  // 只考虑已注册（已估计位姿）的图像
  // 未注册的图像不参与优化，因此不需要为它们选择轨迹
  std::unordered_map<track_t, Track> tracks;
  for (const auto& [image_id, image] : images_) {
    if (!image.IsRegistered()) continue;

    tracks_per_camera[image_id] = 0;  // 初始化轨迹计数为0
  }

  // ========== 步骤3：贪心选择轨迹 ==========
  // cameras_left: 还有多少相机的轨迹数不足
  int cameras_left = tracks_per_camera.size();
  
  // 按长度降序遍历轨迹（优先处理长轨迹）
  for (const auto& [track_length, track_id] : track_lengths) {
    const auto& track = tracks_full.at(track_id);

    // 收集该轨迹涉及的所有注册图像
    // 使用set避免重复（同一图像只计数一次）
    std::unordered_set<image_t> image_ids;
    Track track_temp;  // 临时轨迹对象，只包含注册图像的观测
    for (const auto& [image_id, feature_id] : track.observations) {
      // 跳过未注册的图像
      if (tracks_per_camera.count(image_id) == 0) continue;

      track_temp.track_id = track_id;
      track_temp.observations.emplace_back(
          std::make_pair(image_id, feature_id));
      image_ids.insert(image_id);
    }

    // 过滤条件3：在注册图像中的观测数必须满足最小要求
    if (image_ids.size() < options_.min_num_view_per_track) continue;

    // ===== 决定是否添加该轨迹 =====
    // 策略：如果轨迹的任一观测图像还需要更多轨迹，则添加该轨迹
    bool added = false;  // 标记是否已添加，避免重复插入
    
    for (const auto& [image_id, feature_id] : track_temp.observations) {
      // 获取该图像当前的轨迹数量
      auto& track_per_camera = tracks_per_camera[image_id];
      
      // 如果该图像已有足够的轨迹，跳过
      if (track_per_camera > options_.min_num_tracks_per_view) continue;

      // 否则，增加该图像的轨迹计数
      ++track_per_camera;
      
      // 如果该图像刚好达到最小轨迹数要求，减少需要轨迹的相机数
      if (track_per_camera > options_.min_num_tracks_per_view) --cameras_left;

      // 添加轨迹（只添加一次）
      if (!added) {
        tracks.insert(std::make_pair(track_id, track_temp));
        added = true;
      }
    }
    
    // ===== 终止条件 =====
    // 条件1：所有相机都有足够的轨迹
    if (cameras_left == 0) break;
    // 条件2：已选择的轨迹数达到上限
    if (tracks.size() > options_.max_num_tracks) break;
  }

  // ========== 步骤4：输出选择的轨迹 ==========
  // 复制到输出参数（避免修改tracks_full）
  tracks_selected = tracks;

  return tracks.size();
}

}  // namespace glomap
