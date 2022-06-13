''' utilities for visualization code on KITTI

Date: 2022.06.09

Ref:https://blog.csdn.net/cg129054036/article/details/119516704
'''
import numpy as np
import os
import cv2
from easydict import EasyDict
from PIL import Image
import matplotlib.pyplot as plt
import copy
import open3d as o3d

palette = {'Car': (255 ,0 ,0),
           'Pedestrian': (0, 255, 0),
           'Cyclist': (0, 0, 255)}

class KITTI_Object(object):
    def __init__(self, root='./', split='training', idx='000000'):
        self.idx = idx
        self.image_2_dir = os.path.join(root, split, 'image_2', f'{int(idx):06d}.png')
        self.lidar_dir = os.path.join(root, split, 'velodyne', f'{int(idx):06d}.bin')
        self.label_dir = os.path.join(root, split, 'label_2', f'{int(idx):06d}.txt')
        self.calib_dir = os.path.join(root, split, 'calib', f'{int(idx):06d}.txt')

        with open(self.label_dir, 'r') as f:
            self.meta_data = f.readlines()

    def get_all_objects(self):
        return [Object3D(self.meta_data[i]) for i in range(len(self.meta_data))]

    def get_image(self):
        img = cv2.imread(self.image_2_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_velodyne_lidar(self):
        scan = np.fromfile(self.lidar_dir, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan[:, :3]

    def get_calib(self):
        calib_dict = {}
        with open(self.calib_dir, 'r') as f:
            calib_meta_data = f.readlines()
        for calib_data in calib_meta_data:
            if calib_data == '\n':continue
            key, value = calib_data.rstrip().split(': ')
            value = value.split(' ')
            value = [float(value[i]) for i in range(len(value))]
            value = np.array(value).reshape(3, -1)
            calib_dict[key] = value
        return EasyDict(calib_dict)

    @staticmethod
    def rect_cam0_to_cam2_image(coordinates, P2, return_raw=False):
        """
        :param coordinates: [N, 3]
        :param P:  P2 matrix
        :return:  [N, 2]
        """
        N = len(coordinates)
        coordinates_hom = np.hstack([coordinates, np.ones((N ,1))])
        cam2_pixel = (P2 @ coordinates_hom.T).T
        if return_raw:
            return cam2_pixel
        else:
            cam2_pixel /= cam2_pixel[:, 2:]
            return cam2_pixel[:, :2]

    @staticmethod
    def velodyne_to_cam0(coordinates, velo2cam):
        """
        :param coordinates: [N, 3]
        :param velo2cam: [3, 4]
        :return: [N, 3]
        """
        N = len(coordinates)
        coordinates_hom = np.hstack([coordinates, np.ones((N, 1))])
        return (velo2cam @ coordinates_hom.T).T

    @staticmethod
    def rect_cam0_to_velodyne(coordinates, velo2cam, R0_rect):
        """
        :param coordinates: [N, 3]
        :param velo2cam: [3, 4]
        :param R0_rect: [3, 3]
        :return: [N, 3]
        """
        N = len(coordinates)
        coordinates = (np.linalg.inv(R0_rect) @ coordinates.T).T
        coordinates_hom = np.hstack([coordinates, np.ones((N, 1))])
        velo2cam = np.vstack([velo2cam, [0., 0., 0., 1]])
        return (np.linalg.inv(velo2cam) @ coordinates_hom.T).T[:, :3]

    @staticmethod
    def cam0_to_rect_cam0(coordinates, R0_rect):
        """
        :param coordinates: [N, 3]
        :param R0_rect: [3,3]
        :return:  [N, 3]
        """

        return (R0_rect @ coordinates.T).T

    @staticmethod
    def velodyne_to_cam2_image(coordinates, velo2cam, R0_rect, P2, return_raw=False):
        coordinates = KITTI_Object.velodyne_to_cam0(coordinates, velo2cam)
        coordinates = KITTI_Object.cam0_to_rect_cam0(coordinates, R0_rect)
        return KITTI_Object.rect_cam0_to_cam2_image(coordinates, P2, return_raw)

    @staticmethod
    def get_fov_mask(img, velodyne, velo2cam, R0_rect, P2):
        img2_pixel_from_lidar = KITTI_Object.velodyne_to_cam2_image(velodyne, velo2cam, R0_rect, P2, return_raw=True)
        H, W, _ = img.shape
        depths = np.array(img2_pixel_from_lidar[:, 2])
        img2_pixel_from_lidar /= img2_pixel_from_lidar[:, 2:]
        mask1 = np.logical_and(img2_pixel_from_lidar[:, 0] < W, img2_pixel_from_lidar[:, 1] < H)
        mask2 = np.logical_and(img2_pixel_from_lidar[:, 0] > 0, img2_pixel_from_lidar[:, 1] > 0)
        mask = np.logical_and(mask1, mask2)
        mask = np.logical_and(mask, velodyne[:, 0] > 2.0)
        assert len(mask) == len(velodyne)

        return mask, img2_pixel_from_lidar[:, :2], depths

class Object3D(object):
    def __init__(self, object_info):
        object_info = object_info.rstrip().split(' ')
        self.name = object_info[0]
        object_info[1:] = [float(object_info[i]) for i in range(1, len(object_info))]
        self.truncated = object_info[1]
        self.occluded = object_info[2]
        self.alpha = object_info[3]
        self.xmin = object_info[4]  # left
        self.ymin = object_info[5]  # top
        self.xmax = object_info[6]  # right
        self.ymax = object_info[7]  # bottom
        self.bbox_2d = np.array(object_info[4:8])
        self.h = object_info[8]
        self.w = object_info[9]
        self.l = object_info[10]
        self.location = np.array(object_info[11:14])
        self.rotation_y = object_info[14]

def get_image_with_2d_boxes(img, objects):
    img1 = copy.deepcopy(img)
    for obj in objects:
        if obj.name not in ['Car', 'Pedestrian', 'Cyclist']: continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), palette[obj.name], 2)
        cv2.putText(img1, obj.name.lower(), (int(obj.xmin), int(obj.ymin) - 4), cv2.LINE_AA, 0.75, (0, 0, 255), 2)

    return img1

def get_image_with_3d_boxes(img, calib, objects, color=(255, 255, 255), thickness=2):
    img2 = copy.deepcopy(img)
    for obj in objects:
        if obj.name not in ['Car', 'Pedestrian', 'Cyclist']: continue
        bbox_3d = create_bbox_3d_in_rect_cam0(obj)
        bbox_3d_in_2d_image = KITTI_Object.rect_cam0_to_cam2_image(bbox_3d, P2=calib.P2)  # [8, 2]
        qs = bbox_3d_in_2d_image.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            cv2.line(img2, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img2, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(img2, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return img2

def rotation_y_matrix(ry):
    """
    rotation around y axies, from OZ to OX
    """
    c = np.cos(ry)
    s = np.sin(ry)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def get_image_with_projected_lidar(img, calib, velodyne):
    img3 = copy.deepcopy(img)
    velo_inds, img2_pixel_from_lidar, depths = KITTI_Object.get_fov_mask(img3, velodyne,
                                                                 calib.Tr_velo_to_cam, calib.R0_rect,
                                                                 calib.P2)
    img2_pixel_in_fov = img2_pixel_from_lidar[velo_inds, :]
    depths = depths[velo_inds]
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
    for pixel_in_fov, depth in zip(img2_pixel_in_fov, depths):
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img3, (int(np.round(pixel_in_fov[0])),
                         int(np.round(pixel_in_fov[1]))),
                   2, color=tuple(color), thickness=-1)

    return img3

def show_geometries(geometries=[]):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="kitti")
    vis.get_render_option().point_size = 1
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def create_bbox_3d_in_rect_cam0(obj):
    """
       1 -------- 0
       /|         /|
      2 -------- 3 . H
      | |        | |
      . 5 -------- 4
      |/         |/ W
      6 ---L---- 7

       z
      /
     o ----- x
     |
     y
    """
    roty = rotation_y_matrix(obj.rotation_y)
    h, w, l = obj.h, obj.w, obj.l
    x_corner = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
    y_corner = [-h, -h, -h, -h, 0, 0, 0, 0]
    z_corner = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    bbox_3d = np.vstack([x_corner, y_corner, z_corner])
    bbox_3d = roty @ bbox_3d  # OZ -> OX with rotation_y
    bbox_3d[0, :] += obj.location[0]
    bbox_3d[1, :] += obj.location[1]
    bbox_3d[2, :] += obj.location[2]

    return bbox_3d.T

def show_bboxes_in_lidar(velodyne, calib, objects):
    bboxes = []
    for obj in objects:
        if obj.name not in ['Car', 'Pedestrian', 'Cyclist']: continue
        bbox_3d = create_bbox_3d_in_rect_cam0(obj)
        bbox_3d_in_lidar = KITTI_Object.rect_cam0_to_velodyne(bbox_3d, calib.Tr_velo_to_cam, calib.R0_rect)
        bboxes.append(create_o3d_bbox(bbox_3d_in_lidar))
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(velodyne)
    pcd.paint_uniform_color([1, 1, 1])

    show_geometries([pcd] + bboxes)

def create_o3d_bbox(bbox_3d):
    """
    :param bbox_3d: [8, 3]
    :return: o3d.geometry.Lineset()
    """
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
    colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.points = o3d.utility.Vector3dVector(bbox_3d)

    return line_set

if __name__ == '__main__':
    kitti_object = KITTI_Object(root='./', split='training', idx='000010')
    calib = kitti_object.get_calib()
    velodyne = kitti_object.get_velodyne_lidar()

    objects = kitti_object.get_all_objects()
    print(f"There are {len(objects)} objects.")
    print([obj.name for obj in objects])
    img = kitti_object.get_image()

    img1 = get_image_with_2d_boxes(img, objects)
    img2 = get_image_with_3d_boxes(img, calib, objects)
    img3 = get_image_with_projected_lidar(img, calib, velodyne)
    imgs = np.vstack([img1, img2, img3])
    Image.fromarray(imgs).show()
    
    show_bboxes_in_lidar(velodyne, calib, objects)




