#include "glomap/scene/view_graph.h"

#include "glomap/math/union_find.h"

#include <queue>

namespace glomap {

/**
 * [功能描述]：保留视图图中最大的连通分量，移除孤立的图像集合
 * 连通分量是指通过有效图像对连接在一起的图像集合。保留最大连通分量可以确保重建的连续性
 * @param frames：帧数据集合，将根据连通性更新其注册状态
 * @param images：图像数据集合，用于查询注册状态
 * @return 返回最大连通分量中的图像数量
 */
int ViewGraph::KeepLargestConnectedComponents(
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images) {
  // ========== 步骤1：建立邻接表 ==========
  // 将视图图转换为图论中的邻接表表示
  // 节点：图像，边：有效的图像对
  EstablishAdjacencyList();
  // 建立帧级别的邻接表（考虑图像到帧的映射关系）
  EstablishAdjacencyListFrame(images);

  // ========== 步骤2：查找所有连通分量 ==========
  // 使用DFS或BFS算法找出视图图中的所有连通分量
  // 连通分量：一组通过有效图像对互相连接的图像
  int num_comp = FindConnectedComponent();

  // ========== 步骤3：找到最大的连通分量 ==========
  int max_idx = -1;     // 最大连通分量的索引
  int max_img = 0;      // 最大连通分量的图像数量
  // 遍历所有连通分量，找出包含图像最多的那个
  for (int comp = 0; comp < num_comp; comp++) {
    if (connected_components[comp].size() > max_img) {
      max_img = connected_components[comp].size();
      max_idx = comp;
    }
  }

  // 如果没有有效的连通分量，返回0
  if (max_img == 0) return 0;

  // 获取最大连通分量中的所有图像ID
  std::unordered_set<image_t> largest_component = connected_components[max_idx];

  // ========== 步骤4：更新帧的注册状态 ==========
  // 首先将所有帧标记为未注册
  for (auto& [frame_id, frame] : frames) {
    frame.is_registered = false;
  }
  // 然后将最大连通分量中的帧标记为已注册
  // 只有已注册的帧才会参与后续的重建流程
  for (auto frame_id : largest_component) {
    frames[frame_id].is_registered = true;
  }
  
  // ========== 步骤5：使不在最大连通分量中的图像对失效 ==========
  num_pairs = 0;  // 重新统计有效图像对数量
  for (auto& [pair_id, image_pair] : image_pairs) {
    // 如果图像对的任一图像不在最大连通分量中，标记该图像对为无效
    // 这样做可以：
    // 1. 移除孤立的图像集合，它们无法与主要重建结果连接
    // 2. 确保后续优化（如BA）只处理连通的图像
    // 3. 提高计算效率，避免处理无关的图像对
    if (!images[image_pair.image_id1].IsRegistered() ||
        !images[image_pair.image_id2].IsRegistered()) {
      image_pair.is_valid = false;
    }
    if (image_pair.is_valid) num_pairs++;
  }

  // ========== 步骤6：统计最终的图像数量 ==========
  // 重新计数最大连通分量中的实际图像数量（考虑图像到帧的映射）
  for (auto& [image_id, image] : images) {
    if (image.IsRegistered()) max_img++;
  }
  return max_img;
}

int ViewGraph::FindConnectedComponent() {
  connected_components.clear();
  std::unordered_map<image_t, bool> visited;
  for (auto& [frame_id, neighbors] : adjacency_list_frame) {
    visited[frame_id] = false;
  }

  for (auto& [frame_id, neighbors] : adjacency_list_frame) {
    if (!visited[frame_id]) {
      std::unordered_set<image_t> component;
      BFS(frame_id, visited, component);
      connected_components.push_back(component);
    }
  }

  return connected_components.size();
}

int ViewGraph::MarkConnectedComponents(
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    int min_num_img) {
  EstablishAdjacencyList();
  EstablishAdjacencyListFrame(images);

  int num_comp = FindConnectedComponent();

  std::vector<std::pair<int, int>> cluster_num_img(num_comp);
  for (int comp = 0; comp < num_comp; comp++) {
    cluster_num_img[comp] =
        std::make_pair(connected_components[comp].size(), comp);
  }
  std::sort(cluster_num_img.begin(), cluster_num_img.end(), std::greater<>());

  // Set the cluster number of every frame to be -1
  for (auto& [frame_id, frame] : frames) frame.cluster_id = -1;

  int comp = 0;
  for (; comp < num_comp; comp++) {
    if (cluster_num_img[comp].first < min_num_img) break;
    for (auto frame_id : connected_components[cluster_num_img[comp].second]) {
      frames[frame_id].cluster_id = comp;
    }
  }

  return comp;
}

void ViewGraph::BFS(image_t root,
                    std::unordered_map<image_t, bool>& visited,
                    std::unordered_set<image_t>& component) {
  std::queue<image_t> q;
  q.push(root);
  visited[root] = true;
  component.insert(root);

  while (!q.empty()) {
    image_t curr = q.front();
    q.pop();

    for (image_t neighbor : adjacency_list_frame[curr]) {
      if (!visited[neighbor]) {
        q.push(neighbor);
        visited[neighbor] = true;
        component.insert(neighbor);
      }
    }
  }
}

void ViewGraph::EstablishAdjacencyList() {
  adjacency_list.clear();
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      adjacency_list[image_pair.image_id1].insert(image_pair.image_id2);
      adjacency_list[image_pair.image_id2].insert(image_pair.image_id1);
    }
  }
}

void ViewGraph::EstablishAdjacencyListFrame(
    std::unordered_map<image_t, Image>& images) {
  adjacency_list_frame.clear();
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      frame_t frame_id1 = images[image_pair.image_id1].frame_id;
      frame_t frame_id2 = images[image_pair.image_id2].frame_id;
      adjacency_list_frame[frame_id1].insert(frame_id2);
      adjacency_list_frame[frame_id2].insert(frame_id1);
    }
  }
}
}  // namespace glomap
