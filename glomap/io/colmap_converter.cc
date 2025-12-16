#include "glomap/io/colmap_converter.h"

#include "glomap/math/two_view_geometry.h"

#include "colmap/scene/reconstruction_io_utils.h"

namespace glomap {

void ConvertGlomapToColmapImage(const Image& image,
                                colmap::Image& image_colmap,
                                bool keep_points) {
  image_colmap.SetImageId(image.image_id);
  image_colmap.SetCameraId(image.camera_id);
  image_colmap.SetName(image.file_name);
  image_colmap.SetFrameId(image.frame_id);

  if (keep_points) {
    image_colmap.SetPoints2D(image.features);
  }
}

/**
 * [功能描述]：将Glomap格式的重建数据转换为COLMAP格式
 * 该函数将Glomap的相机、帧、图像、3D轨迹等数据结构转换为COLMAP的Reconstruction对象。
 * 支持聚类簇过滤，可以只转换指定簇的数据。
 * 
 * @param rigs：相机装备映射表，键为装备ID，值为Rig对象
 * @param cameras：相机参数映射表，键为相机ID，值为Camera对象
 * @param frames：帧数据映射表，键为帧ID，值为Frame对象
 * @param images：图像数据映射表，键为图像ID，值为Image对象
 * @param tracks：3D点轨迹映射表，键为轨迹ID，值为Track对象
 * @param reconstruction：输出的COLMAP重建对象（引用传递，函数会修改此对象）
 * @param cluster_id：聚类簇ID，-1表示转换所有数据，否则只转换指定簇的数据，默认值为-1
 * @param include_image_points：是否包含图像特征点，默认值为false
 * @return 无返回值
 */
void ConvertGlomapToColmap(const std::unordered_map<rig_t, Rig>& rigs,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<frame_t, Frame>& frames,
                           const std::unordered_map<image_t, Image>& images,
                           const std::unordered_map<track_t, Track>& tracks,
                           colmap::Reconstruction& reconstruction,
                           int cluster_id,
                           bool include_image_points) {
  // 清空COLMAP重建对象，确保从空白状态开始
  reconstruction = colmap::Reconstruction();

  // 第一步：添加所有相机参数到COLMAP重建对象
  // 相机参数包括焦距、畸变参数等内参信息
  for (const auto& [camera_id, camera] : cameras) {
    reconstruction.AddCamera(camera);
  }

  // 第二步：添加相机装备（rigs）
  // Rig表示多个相机的刚性组合，例如立体相机或多相机阵列
  for (const auto& [rig_id, rig] : rigs) {
    reconstruction.AddRig(rig);
  }

  // 第三步：添加帧数据
  for (auto& [frame_id, frame] : frames) {
    Frame frame_curr = frame;  // 复制帧对象以避免悬空指针
    frame_curr.ResetRigPtr();  // 重置rig指针，避免指针失效
    reconstruction.AddFrame(frame_curr);
  }

  // 第四步：准备2D-3D对应关系
  // min_supports定义了一个3D点至少需要在多少张图像中被观测到
  size_t min_supports = 2;
  // image_to_point3D存储每张图像的特征点对应的3D轨迹ID
  // 结构：image_id -> [特征点索引对应的track_id数组]
  std::unordered_map<image_t, std::vector<track_t>> image_to_point3D;
  
  if (tracks.size() > 0 || include_image_points) {
    // 初始化每张图像的特征点，默认都对应到无效的3D点（-1）
    for (auto& [image_id, image] : images) {
      // 跳过未注册的图像或不属于指定聚类簇的图像
      if (!image.IsRegistered() ||
          (cluster_id != -1 && image.ClusterId() != cluster_id))
        continue;
      // 为每个特征点创建一个track_id槽位，初始值为-1（表示无对应3D点）
      image_to_point3D[image_id] =
          std::vector<track_t>(image.features.size(), -1);
    }

    // 如果存在3D轨迹数据，则建立2D特征点到3D轨迹的映射关系
    if (tracks.size() > 0) {
      for (auto& [track_id, track] : tracks) {
        // 跳过观测数量不足的轨迹（至少需要2个观测）
        if (track.observations.size() < min_supports) {
          continue;
        }
        // 遍历该轨迹的所有观测
        // observation.first是image_id，observation.second是特征点索引
        for (auto& observation : track.observations) {
          if (image_to_point3D.find(observation.first) !=
              image_to_point3D.end()) {
            // 将该图像的对应特征点索引关联到当前track_id
            image_to_point3D[observation.first][observation.second] = track_id;
          }
        }
      }
    }
  }

  // 第五步：添加3D点到COLMAP重建对象
  for (const auto& [track_id, track] : tracks) {
    // 创建COLMAP格式的3D点
    colmap::Point3D colmap_point;
    colmap_point.xyz = track.xyz;      // 3D点的世界坐标
    colmap_point.color = track.color;  // 3D点的RGB颜色
    colmap_point.error = 0;            // 初始化重投影误差为0

    // 添加该3D点的所有观测（track elements）
    for (auto& observation : track.observations) {
      const Image& image = images.at(observation.first);
      // 跳过未注册的图像或不属于指定聚类簇的图像
      if (!image.IsRegistered() ||
          (cluster_id != -1 && image.ClusterId() != cluster_id))
        continue;
      
      // 创建COLMAP格式的轨迹元素
      colmap::TrackElement colmap_track_el;
      colmap_track_el.image_id = observation.first;      // 观测到该点的图像ID
      colmap_track_el.point2D_idx = observation.second;  // 该点在图像中的2D特征点索引

      colmap_point.track.AddElement(colmap_track_el);
    }

    // 如果轨迹长度（观测数量）小于最小支持数，则跳过该点
    if (colmap_point.track.Length() < min_supports) continue;

    // 压缩轨迹数据以节省内存
    colmap_point.track.Compress();
    // 将3D点添加到重建对象中
    reconstruction.AddPoint3D(track_id, std::move(colmap_point));
  }

  // 第六步：添加图像到COLMAP重建对象
  for (const auto& [image_id, image] : images) {
    colmap::Image image_colmap;
    // 检查该图像是否需要保留特征点信息
    bool keep_points =
        image_to_point3D.find(image_id) != image_to_point3D.end();
    
    // 将Glomap格式的图像转换为COLMAP格式
    ConvertGlomapToColmapImage(image, image_colmap, keep_points);
    
    // 如果需要保留特征点，则设置2D特征点到3D点的对应关系
    if (keep_points) {
      std::vector<track_t>& track_ids = image_to_point3D[image_id];
      // 遍历图像的所有特征点
      for (size_t i = 0; i < image.features.size(); i++) {
        // 如果该特征点有对应的3D点，且该3D点存在于重建中
        if (track_ids[i] != -1 && reconstruction.ExistsPoint3D(track_ids[i])) {
          // 建立2D特征点到3D点的关联
          image_colmap.SetPoint3DForPoint2D(i, track_ids[i]);
        }
      }
    }

    // 将图像添加到重建对象中
    reconstruction.AddImage(std::move(image_colmap));
  }

  // 第七步：注销不需要的帧
  // 如果帧未注册或不属于指定的聚类簇，则从重建中移除
  for (auto& [frame_id, frame] : frames) {
    if ((cluster_id != 0 && !frame.is_registered) ||
        (frame.cluster_id != cluster_id && cluster_id != -1)) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }

  // 第八步：更新所有3D点的重投影误差
  reconstruction.UpdatePoint3DErrors();
}

void ConvertColmapToGlomap(const colmap::Reconstruction& reconstruction,
                           std::unordered_map<rig_t, Rig>& rigs,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<frame_t, Frame>& frames,
                           std::unordered_map<image_t, Image>& images,
                           std::unordered_map<track_t, Track>& tracks) {
  // Clear the glomap reconstruction
  cameras.clear();
  images.clear();

  // Add cameras
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    cameras[camera_id] = camera;
  }

  // Add rigs
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    rigs[rig_id] = rig;
  }

  // Add frames
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    frames[frame_id] = frame;
    frames[frame_id].SetRigPtr(rigs.find(frame.RigId()) != rigs.end()
                                   ? &rigs[frame.RigId()]
                                   : nullptr);
    frames[frame_id].is_registered = frame.HasPose();
  }

  for (auto& [image_id, image_colmap] : reconstruction.Images()) {
    auto ite = images.insert(std::make_pair(image_colmap.ImageId(),
                                            Image(image_colmap.ImageId(),
                                                  image_colmap.CameraId(),
                                                  image_colmap.Name())));

    Image& image = ite.first->second;
    image.frame_id = image_colmap.FrameId();
    image.frame_ptr = frames.find(image.frame_id) != frames.end()
                          ? &frames[image.frame_id]
                          : nullptr;
    image.features.clear();
    image.features.reserve(image_colmap.NumPoints2D());

    for (auto& point2D : image_colmap.Points2D()) {
      image.features.push_back(point2D.xy);
    }
  }

  ConvertColmapPoints3DToGlomapTracks(reconstruction, tracks);
}

void ConvertColmapPoints3DToGlomapTracks(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<track_t, Track>& tracks) {
  // Read tracks
  tracks.clear();
  tracks.reserve(reconstruction.NumPoints3D());

  auto& points3D = reconstruction.Points3D();
  for (auto& [point3d_id, point3D] : points3D) {
    Track track;
    const colmap::Track& track_colmap = point3D.track;
    track.xyz = point3D.xyz;
    track.color = point3D.color;
    track.track_id = point3d_id;
    track.is_initialized = true;

    const std::vector<colmap::TrackElement>& elements = track_colmap.Elements();
    track.observations.reserve(track_colmap.Length());
    for (auto& element : elements) {
      track.observations.push_back(
          Observation(element.image_id, element.point2D_idx));
    }

    tracks.insert(std::make_pair(point3d_id, track));
  }
}

// For ease of debug, go through the database twice: first extract the
// available pairs, then read matches from pairs.
void ConvertDatabaseToGlomap(const colmap::Database& database,
                             ViewGraph& view_graph,
                             std::unordered_map<rig_t, Rig>& rigs,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<frame_t, Frame>& frames,
                             std::unordered_map<image_t, Image>& images) {
  // Add the images
  std::vector<colmap::Image> images_colmap = database.ReadAllImages();
  image_t counter = 0;
  for (auto& image : images_colmap) {
    std::cout << "\r Loading Images " << counter + 1 << " / "
              << images_colmap.size() << std::flush;
    counter++;

    const image_t image_id = image.ImageId();
    if (image_id == colmap::kInvalidImageId) continue;
    images.insert(std::make_pair(
        image_id, Image(image_id, image.CameraId(), image.Name())));

    // TODO: Implement the logic of reading prior pose from the database
    // const colmap::PosePrior prior = database.ReadPosePrior(image_id);
    // if (prior.IsValid()) {
    //   const colmap::Rigid3d
    //   world_from_cam_prior(Eigen::Quaterniond::Identity(),
    //                                              prior.position);
    //   ite.first->second.cam_from_world =
    //   Rigid3d(Inverse(world_from_cam_prior));
    // } else {
    //   ite.first->second.cam_from_world = Rigid3d();
    // }
  }
  std::cout << std::endl;

  // Read keypoints
  for (auto& [image_id, image] : images) {
    const colmap::FeatureKeypoints keypoints = database.ReadKeypoints(image_id);
    const int num_keypoints = keypoints.size();
    image.features.resize(num_keypoints);
    for (int i = 0; i < num_keypoints; i++) {
      image.features[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
    }
  }

  // Add the cameras
  std::vector<colmap::Camera> cameras_colmap = database.ReadAllCameras();
  for (auto& camera : cameras_colmap) {
    cameras[camera.camera_id] = camera;
  }

  // Add the rigs
  std::vector<colmap::Rig> rigs_colmap = database.ReadAllRigs();
  for (auto& rig : rigs_colmap) {
    rigs[rig.RigId()] = rig;
  }

  // Add the frames
  std::vector<colmap::Frame> frames_colmap = database.ReadAllFrames();
  for (auto& frame : frames_colmap) {
    frame_t frame_id = frame.FrameId();
    if (frame_id == colmap::kInvalidFrameId) continue;
    frames[frame_id] = Frame(frame);
    frames[frame_id].SetRigId(frame.RigId());
    frames[frame_id].SetRigPtr(rigs.find(frame.RigId()) != rigs.end()
                                   ? &rigs[frame.RigId()]
                                   : nullptr);
    frames[frame_id].SetRigFromWorld(Rigid3d());

    for (auto data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) != images.end()) {
        images[image_id].frame_id = frame_id;
        images[image_id].frame_ptr = &frames[frame_id];
      }
    }
  }

  // cameras that are not used in any rig
  rig_t max_rig_id = 0;
  std::unordered_map<camera_t, rig_t> cameras_id_to_rig_id;
  for (const auto& [rig_id, rig] : rigs) {
    max_rig_id = std::max(max_rig_id, rig_id);

    sensor_t sensor_id = rig.RefSensorId();
    if (sensor_id.type == SensorType::CAMERA) {
      cameras_id_to_rig_id[rig.RefSensorId().id] = rig_id;
    }
    const std::map<sensor_t, std::optional<Rigid3d>>& sensors =
        rig.NonRefSensors();
    for (const auto& [sensor_id, sensor_pose] : sensors) {
      if (sensor_id.type == SensorType::CAMERA) {
        cameras_id_to_rig_id[sensor_id.id] = rig_id;
      }
    }
  }

  // For cameras that are not in any rig, add camera rigs
  for (const auto& [camera_id, camera] : cameras) {
    if (cameras_id_to_rig_id.find(camera_id) == cameras_id_to_rig_id.end()) {
      Rig rig;
      rig.SetRigId(++max_rig_id);
      rig.AddRefSensor(camera.SensorId());
      rigs[rig.RigId()] = rig;
      cameras_id_to_rig_id[camera_id] = rig.RigId();
    }
  }

  frame_t max_frame_id = 0;
  // For frames that are not in any rig, add camera rigs
  for (const auto& [frame_id, frame] : frames) {
    if (frame_id == colmap::kInvalidFrameId) continue;
    max_frame_id = std::max(max_frame_id, frame_id);
  }

  // For images without frames, initialize trivial frames
  for (auto& [image_id, image] : images) {
    if (image.frame_id == colmap::kInvalidFrameId) {
      frame_t frame_id = ++max_frame_id;

      CreateFrameForImage(Rigid3d(),
                          image,
                          rigs,
                          frames,
                          cameras_id_to_rig_id[image.camera_id],
                          frame_id);
    }
  }

  // Add the matches
  std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>>
      all_matches = database.ReadAllMatches();

  // Go through all matches and store the matche with enough observations in
  // the view_graph
  size_t invalid_count = 0;
  std::unordered_map<image_pair_t, ImagePair>& image_pairs =
      view_graph.image_pairs;
  for (size_t match_idx = 0; match_idx < all_matches.size(); match_idx++) {
    if ((match_idx + 1) % 1000 == 0 || match_idx == all_matches.size() - 1)
      std::cout << "\r Loading Image Pair " << match_idx + 1 << " / "
                << all_matches.size() << std::flush;
    // Read the image pair from COLMAP database
    colmap::image_pair_t pair_id = all_matches[match_idx].first;
    std::pair<colmap::image_t, colmap::image_t> image_pair_colmap =
        colmap::PairIdToImagePair(pair_id);
    colmap::image_t image_id1 = image_pair_colmap.first;
    colmap::image_t image_id2 = image_pair_colmap.second;

    colmap::FeatureMatches& feature_matches = all_matches[match_idx].second;

    // Initialize the image pair
    auto ite = image_pairs.insert(
        std::make_pair(ImagePair::ImagePairToPairId(image_id1, image_id2),
                       ImagePair(image_id1, image_id2)));
    ImagePair& image_pair = ite.first->second;

    colmap::TwoViewGeometry two_view =
        database.ReadTwoViewGeometry(image_id1, image_id2);

    // If the image is marked as invalid or watermark, then skip
    if (two_view.config == colmap::TwoViewGeometry::UNDEFINED ||
        two_view.config == colmap::TwoViewGeometry::DEGENERATE ||
        two_view.config == colmap::TwoViewGeometry::WATERMARK ||
        two_view.config == colmap::TwoViewGeometry::MULTIPLE) {
      image_pair.is_valid = false;
      invalid_count++;
      continue;
    }

    // Collect the fundemental matrices
    if (two_view.config == colmap::TwoViewGeometry::UNCALIBRATED) {
      image_pair.F = two_view.F;
    } else if (two_view.config == colmap::TwoViewGeometry::CALIBRATED) {
      FundamentalFromMotionAndCameras(
          cameras.at(images.at(image_pair.image_id1).camera_id),
          cameras.at(images.at(image_pair.image_id2).camera_id),
          two_view.cam2_from_cam1,
          &image_pair.F);
    } else if (two_view.config == colmap::TwoViewGeometry::PLANAR ||
               two_view.config == colmap::TwoViewGeometry::PANORAMIC ||
               two_view.config ==
                   colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC) {
      image_pair.H = two_view.H;
      image_pair.F = two_view.F;
    }
    image_pair.config = two_view.config;

    // Collect the matches
    image_pair.matches = Eigen::MatrixXi(feature_matches.size(), 2);

    std::vector<Eigen::Vector2d>& keypoints1 =
        images[image_pair.image_id1].features;
    std::vector<Eigen::Vector2d>& keypoints2 =
        images[image_pair.image_id2].features;

    feature_t count = 0;
    for (int i = 0; i < feature_matches.size(); i++) {
      colmap::point2D_t point2D_idx1 = feature_matches[i].point2D_idx1;
      colmap::point2D_t point2D_idx2 = feature_matches[i].point2D_idx2;
      if (point2D_idx1 != colmap::kInvalidPoint2DIdx &&
          point2D_idx2 != colmap::kInvalidPoint2DIdx) {
        if (keypoints1.size() <= point2D_idx1 ||
            keypoints2.size() <= point2D_idx2)
          continue;
        image_pair.matches.row(count) << point2D_idx1, point2D_idx2;
        count++;
      }
    }
    image_pair.matches.conservativeResize(count, 2);
  }
  std::cout << std::endl;

  LOG(INFO) << "Pairs read done. " << invalid_count << " / "
            << view_graph.image_pairs.size() << " are invalid";
}

void CreateOneRigPerCamera(const std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<rig_t, Rig>& rigs) {
  for (const auto& [camera_id, camera] : cameras) {
    Rig rig;
    rig.SetRigId(camera_id);
    rig.AddRefSensor(camera.SensorId());
  }
}

void CreateFrameForImage(const Rigid3d& cam_from_world,
                         Image& image,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<frame_t, Frame>& frames,
                         rig_t rig_id,
                         frame_t frame_id) {
  Frame frame;
  if (frame_id == colmap::kInvalidFrameId) {
    frame_id = image.image_id;
  }
  if (rig_id == colmap::kInvalidRigId) {
    rig_id = image.camera_id;
  }
  frame.SetFrameId(frame_id);
  frame.SetRigId(rig_id);
  frame.SetRigPtr(rigs.find(rig_id) != rigs.end() ? &rigs[rig_id] : nullptr);
  frame.AddDataId(image.DataId());
  frame.SetRigFromWorld(cam_from_world);
  frames[frame_id] = frame;

  image.frame_id = frame_id;
  image.frame_ptr = &frames[frame_id];
}

}  // namespace glomap
