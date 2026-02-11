"""
Author: Redal
Date: 2026-01-28
Todo: 实现TransFuser自动驾驶模型的输入特征构建器TransfuserFeatureBuilder:
      a.拼接并缩放前视三摄像头图像(左、前、右)为一张宽幅RGB图;b.将LiDAR点云
      按高度分层,投影到鸟瞰图BEV生成2D直方图作为LiDAR特征;c.拼接自车当前状
      态(驾驶指令、速度、加速度)作为状态特征.
      输出目标构建器TransfuserTargetBuilder:a.提取未来轨迹用于自车规划;b.在
      BEV空间中提取邻近交通参与者车辆的2D有向包围框及其存在标签;c.渲染包含车
      道线、道路边界、交通参与者等语义信息的 多类别BEV语义地图,用于辅助感知与规划.
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import torch
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, MapObject, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import LineString, Polygon
from torchvision import transforms
# Import local files from customized package
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Input features builder for Transfuser agent"""
    def __init__(self, config: TransfuserConfig):
        """Initializes feature builder
        :param config: global config dataclass of TransFuser"""
        self._config = config
    def get_unique_name(self,)->str:
        """Inherited, see superclass"""
        return "transfuser_feature"
    def compute_features(self, agent_input: AgentInput
                         )->Dict[str, torch.Tensor]:
        """Build the input features for camera/lidar/status features
        for TransFuser agent to processing data and extracting targets"""
        features = {}
        features["camera_feature"] = self._get_camera_feature(agent_input)
        if not self._config.use_lidar: 
            features["lidar_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate([
            torch.tensor(agent_input.ego_status[-1].driving_command, dtype=torch.float32),
            torch.tensor(agent_input.ego_status[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(agent_input.ego_status[-1].ego_acceleration, dtype=torch.float32)])
        return features
    def _get_camera_feature(self, agent_input: AgentInput)->torch.Tensor:
        """Extract stitched camera from AgentInput
        agent_input: input dataclass, stitched front view image as torch tensor"""
        cameras = agent_input.cameras[-1]
        # crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]
        # stitch images l0, r0, f0 together
        stitched_image = np.concatenate([l0, f0, r0],axis=1)
        # resize the stitched image and convert to torch tensor
        resized_image = cv2.resize(stitched_image, (2048, 512))
        tensor_image = transforms.ToTensor()(resized_image)
        return tensor_image
    def _get_lidar_feature(self, agent_input: AgentInput)->torch.Tensor:
        """Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors"""
        # for lidar, only consider (x, y, z) & swap axes for (N, 3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
        # design splat_points function to create histogram
        def splat_points(point_cloud):
            # use 256x256 grid
            xbins = np.linspace(self._config.lidar_min_x,
                                self._config.lidar_max_x,
                                (self._config.lidar_max_x - self._config.lidar_min_x)*int(self._config.pixels_per_meter) + 1)
            ybins = np.linspace(self._config.lidar_min_y,
                                self._config.lidar_max_y,
                                (self._config.lidar_max_y - self._config.lidar_min_y)*int(self._config.pixels_per_meter) + 1)
            # compute histogram by point_cloud and xbins/ybins
            hist = np.histogram2d(point_cloud[:, :2], bins=[xbins, ybins])[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            # make normalization for histogram data
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat
        # remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        # if activating use_ground_plane, use below points
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        # convert features shape as [C, H, W] to output torch tensor
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return torch.tensor(features)
    

class TransfuserTargetBuilder(AbstractTargetBuilder):
    """Ouput Target Builder for TransFuser"""
    def __init__(self, trajectory_sampling: TrajectorySampling,
                 config: TransfuserConfig):
        """Initializes the TargetBuilder
        trajectory_sampling: trajectory sampling specification
        config: global config dataclass of TransFuser"""
        self._trajectory_sampling = trajectory_sampling
        self._config = config
    def get_unique_name(self,)->str:
        """Inherited, see superclass"""
        return "transfuser_target_builder"
    def compute_targets(self, scene: Scene)->Dict[str, torch.Tensor]:
        """Calculate the target data for the given scenario, including 
        trajectories, agent states, agent labels, and BEV semantic maps
        trajectory: the future trajectory of the agent, shape [num_poses, ...];
        agent_states: the states of the agent;"""
        trajectory = torch.tensor(scene.get_future_trajectory(
                num_trajectory_frames=self._trajectory_sampling.num_poses).poses)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        # Extract the annotation information of the current frame and the ego-vehicle's posture
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
        # Compute the ego_agent states and labels
        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)
        return {'trajectory': trajectory,
                'agent_states': agent_states,
                'agent_labels': agent_labels,
                'bev_semantic_map': bev_semantic_map,}
    def _compute_agent_targets(self, annotations: Annotations
                        )->Tuple[torch.Tensor, torch.Tensor]:
        """Extract the 2D bounding box information of the vehicle from the annotation data, and 
        return the bounding box values in tensor form and binary labels"""
        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []
        def _xy_in_lidar(x:float, y:float, config:TransfuserConfig)->bool:
            """Extract the 2D bounding box information of the vehicle from the 
            annotation data, and return the bounding box values in tensor form and binary labels"""
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)
        # Iterate through all bounding boxes and names, and filter out vehicle agents that meet the criteria
        # the agent states contain the x/y/heading/length/width
        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],)
            # Only consider the name "vehicle" and check the center whether is the bounding box
            if name=='vehicle' and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array(
                    [box_x, box_y, box_heading, box_length, box_width], dtype=np.float32))
        agents_states_arr = np.array(agent_states_list)
        # Initialize the agent states and annotations for output
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        # 
        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBoxIndex.POINT2D], axis=-1)
            argsort = np.argsort(distances)[:max_agents]
            # filter the detections in the cameras and lidar
            agents_states_arr = agents_states_arr[argsort]
            agent_states[:, len(agents_states_arr)] = agents_states_arr
            agent_states[: len(agents_states_arr)] = True
        return torch.tensor(agent_states), torch.tensor(agent_labels)
    def _compute_bev_semantic_map(self, annotations: Annotations, 
                                  map_api: AbstractMap,
                                  ego_pose: StateSE2
                                  )->torch.Tensor:
        """Create semantic map in the BEV space, annotations: Annotation classes
        map_api: Map interface of nuPlan used to access map elements
        ego_pose: Ego vehicle pose in the global coordinate frame"""
        # Initialize an empty BEV map for zemantic map with zeros
        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        # Iterate through all the bev semantic classes defined in the config
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type=='polygon':
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type=='linestring':
                entity_mask = self._compute_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            # Assign labels to positions in the BEV map where the entity_mask is
            bev_semantic_map[entity_mask] = label
        # convert the numpy ndarray to torch tensor
        return torch.tensor(bev_semantic_map)
    def _compute_map_polygon_mask(self, map_api: AbstractMap,
                                  ego_pose: StateSE2,
                                  layers: List[SemanticMapLayer]
                                  )->npt.NDArray[np.bool_]:
        """Compute binary mask given a map layer class
        map_api: map interface of nuPlan; ego_pose: ego pose in the map frame; 
        layers: map layers; Returns: binary mask as numpy array"""
        # Get map objects within a specified radius around the ego vehicle
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers)
        # Initialize a zero mask in the shape of the BEV map
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            # iterate through each layer of the map_object_dict
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.geometry, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # opencv has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0
    def _compute_map_linestring_mask(self, map_api: AbstractMap,
                                     ego_pose: StateSE2, 
                                     layers: List[SemanticMapLayer],
                                     )->npt.NDArray[np.bool_]:
        """compute binary of linestring mask given a map layer class
        map_api: map interface of nuPlam; ego_pose: ego pose in global frame
        layers: list of map layers; Returns: binary mask as numpy array"""
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers)
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, 
                              [points],
                              isClosed=False,
                              color=255,
                              thickness=2)
        # Opence has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0
    def _compute_box_mask(self, annotations: Annotations, 
                          layers: TrackedObjectType,
                          )->npt.NDArray[np.bool_]:
        """"""
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                

        

            





