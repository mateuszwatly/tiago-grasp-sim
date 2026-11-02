import math

import numpy as np


PAPER_DOI = "10.1177/1729881419831846"


def vector_list(vector):
    return [float(value) for value in vector]


def pose_dict(position, yaw):
    return {
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "yaw": float(yaw),
    }


class GeometryContactPolicy:
    """Two-contact geometric grasp heuristic adapted from Zapata-Impata et al."""

    def __init__(
        self,
        random_generator,
        fingertip_width=0.018,
        gripper_max_width=0.09,
        pregrasp_distance=0.10,
        lift_distance=0.05,
        finger_pad_z_correction=-0.012,
        finger_pad_forward_offset=0.045,
        finger_pad_half_height=0.025,
        gripper_body_vertical_half_extent=0.039,
    ):
        self.random = random_generator
        self.fingertip_width = float(fingertip_width)
        self.gripper_max_width = float(gripper_max_width)
        self.pregrasp_distance = float(pregrasp_distance)
        self.lift_distance = float(lift_distance)
        self.finger_pad_z_correction = float(finger_pad_z_correction)
        self.finger_pad_forward_offset = float(finger_pad_forward_offset)
        self.finger_pad_half_height = float(finger_pad_half_height)
        self.gripper_body_vertical_half_extent = float(
            gripper_body_vertical_half_extent
        )

    def compute(self, object_info):
        points, normals, curvature = self.sample_cloud(object_info)
        centroid = points.mean(axis=0)
        covariance = np.cov((points - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if main_axis[2] < 0.0:
            main_axis = -main_axis

        approach = np.array([centroid[0], centroid[1], 0.0], dtype=float)
        approach /= max(np.linalg.norm(approach), 1e-9)
        closing_axis = np.cross(np.array([0.0, 0.0, 1.0]), approach)
        closing_axis /= max(np.linalg.norm(closing_axis), 1e-9)

        plane_distance = np.abs((points - centroid) @ main_axis)
        object_center_z = (
            float(object_info["table_height"])
            + float(object_info["geometry"]["center_height"])
        )
        grasp_plane_z = max(
            object_center_z + self.finger_pad_z_correction,
            float(object_info["table_height"])
            + self.gripper_body_vertical_half_extent
            + 0.026,
        )
        # The PAL gripper closes in the horizontal plane. Restrict both
        # contacts to an equatorial slice so a sphere cannot yield a diagonal
        # top-to-bottom pair that the two fingers cannot realize.
        cutting_indices = np.flatnonzero(
            np.abs(points[:, 2] - grasp_plane_z) <= 0.007
        )
        if cutting_indices.size < 40:
            cutting_indices = np.argsort(
                np.abs(points[:, 2] - grasp_plane_z)
            )[:40]
        cutting_points = points[cutting_indices]
        projection = (cutting_points - centroid) @ closing_axis
        first_seed = cutting_points[int(np.argmin(projection))]
        second_seed = cutting_points[int(np.argmax(projection))]
        object_width = float(np.linalg.norm(second_seed - first_seed))
        if object_width > self.gripper_max_width:
            raise RuntimeError(
                f"Object width {object_width:.4f} m exceeds gripper width "
                f"{self.gripper_max_width:.4f} m"
            )

        region_radius = 2.0 * self.fingertip_width
        if object_width <= 2.0 * region_radius:
            region_radius = object_width * 0.45
        first_region = self.region_indices(points, first_seed, region_radius)
        second_region = self.region_indices(points, second_seed, region_radius)
        first_region = self.voxelize(points, first_region, self.fingertip_width * 0.5)
        second_region = self.voxelize(points, second_region, self.fingertip_width * 0.5)

        first_plane_scale = self.scale_values(plane_distance[first_region])
        second_plane_scale = self.scale_values(plane_distance[second_region])
        first_curve_scale = self.scale_values(curvature[first_region])
        second_curve_scale = self.scale_values(curvature[second_region])

        best = None
        for first_local, first_index in enumerate(first_region):
            for second_local, second_index in enumerate(second_region):
                first = points[first_index]
                second = points[second_index]
                connector = second - first
                separation = float(np.linalg.norm(connector))
                if separation <= 1e-6 or separation > self.gripper_max_width:
                    continue
                if (
                    abs(float(first[2] - grasp_plane_z)) > 0.009
                    or abs(float(second[2] - grasp_plane_z)) > 0.009
                    or abs(float(connector[2])) > 0.008
                ):
                    continue
                connector /= separation
                antipodal_first = abs(float(np.dot(normals[first_index], connector)))
                antipodal_second = abs(float(np.dot(normals[second_index], connector)))
                perpendicular = 1.0 - abs(float(np.dot(connector, main_axis)))
                r1 = (
                    2.0
                    - float(first_plane_scale[first_local])
                    - float(second_plane_scale[second_local])
                    + 2.0 * perpendicular
                )
                r2 = (
                    2.0
                    - float(first_curve_scale[first_local])
                    - float(second_curve_scale[second_local])
                    + antipodal_first
                    + antipodal_second
                    - abs(antipodal_first - antipodal_second)
                )
                score = 1.5 * r1 + r2
                if best is None or score > best[0]:
                    best = (score, first_index, second_index, separation)
        if best is None:
            raise RuntimeError("No reachable antipodal contact pair was found")

        score, first_index, second_index, separation = best
        first_contact = points[first_index]
        second_contact = points[second_index]
        midpoint = 0.5 * (first_contact + second_contact)
        target = midpoint.copy()
        # The PAL grasping frame is at the rear of the 110 mm finger collision
        # boxes. Put the contacts at the pad center rather than at that frame.
        target -= approach * self.finger_pad_forward_offset
        # Use the object's physical centre rather than the height of a random
        # sampled contact pair.  Dynamic objects settle from their 3 mm spawn
        # clearance onto this exact height before an episode begins.
        target[2] = grasp_plane_z
        # Descend onto the side-grasp pose from above. Sweeping the PAL
        # gripper's 110 mm finger boxes horizontally can push tall cups and
        # cans before they enter the open finger gap.
        pregrasp = target.copy()
        pregrasp[2] += self.pregrasp_distance
        lift = target.copy()
        lift[2] += self.lift_distance
        yaw = math.atan2(float(approach[1]), float(approach[0]))
        gripper_contact_width = separation
        if object_info["geometry"]["shape"] == "sphere":
            # The 50 mm-tall pads overlap the sphere's equator even though the
            # selected contact points lie in the elevated collision-free
            # grasp plane. The closing command must therefore use the full
            # sphere diameter.
            gripper_contact_width = 2.0 * float(
                object_info["geometry"]["half_width"]
            )

        return {
            "name": "pca_antipodal_two_finger",
            "reference_doi": PAPER_DOI,
            "point_count": int(points.shape[0]),
            "centroid": vector_list(centroid),
            "pca_eigenvalues": vector_list(eigenvalues),
            "main_axis": vector_list(main_axis),
            "approach_axis": vector_list(approach),
            "closing_axis": vector_list(closing_axis),
            "contacts": [vector_list(first_contact), vector_list(second_contact)],
            "contact_normals": [
                vector_list(normals[first_index]),
                vector_list(normals[second_index]),
            ],
            "contact_separation": float(gripper_contact_width),
            "sampled_contact_separation": float(separation),
            "rank_score": float(score),
            "pregrasp_pose": pose_dict(pregrasp, yaw),
            "grasp_pose": pose_dict(target, yaw),
            "lift_pose": pose_dict(lift, yaw),
            "parameters": {
                "fingertip_width": self.fingertip_width,
                "gripper_max_width": self.gripper_max_width,
                "cutting_plane_band": 0.007,
                "pregrasp_distance": self.pregrasp_distance,
                "lift_distance": self.lift_distance,
                "finger_pad_z_correction": self.finger_pad_z_correction,
                "finger_pad_forward_offset": self.finger_pad_forward_offset,
                "finger_pad_half_height": self.finger_pad_half_height,
                "gripper_body_vertical_half_extent": self.gripper_body_vertical_half_extent,
            },
        }

    def sample_cloud(self, object_info):
        geometry = object_info["geometry"]
        center = np.array(
            [
                object_info["pose"]["x"],
                object_info["pose"]["y"],
                object_info["pose"]["z"],
            ],
            dtype=float,
        )
        radius = float(geometry["half_width"])
        count = 720
        if geometry["shape"] == "sphere":
            u = np.array([self.random.uniform(-1.0, 1.0) for _ in range(count)])
            theta = np.array(
                [self.random.uniform(-math.pi, math.pi) for _ in range(count)]
            )
            radial = np.sqrt(np.maximum(0.0, 1.0 - u * u))
            normals = np.column_stack(
                (radial * np.cos(theta), radial * np.sin(theta), u)
            )
            points = center + radius * normals
            curvature = np.full(count, 1.0 / radius, dtype=float)
            return points, normals, curvature

        half_height = float(geometry["center_height"])
        theta = np.array(
            [self.random.uniform(-math.pi, math.pi) for _ in range(count)]
        )
        heights = np.array(
            [self.random.uniform(-half_height, half_height) for _ in range(count)]
        )
        normals = np.column_stack(
            (np.cos(theta), np.sin(theta), np.zeros(count, dtype=float))
        )
        points = center + np.column_stack(
            (radius * np.cos(theta), radius * np.sin(theta), heights)
        )
        curvature = np.full(count, 0.5 / radius, dtype=float)
        return points, normals, curvature

    @staticmethod
    def region_indices(points, seed, radius):
        indices = np.flatnonzero(np.linalg.norm(points - seed, axis=1) <= radius)
        if indices.size == 0:
            raise RuntimeError("A geometric contact region was empty")
        return indices

    @staticmethod
    def voxelize(points, indices, spacing):
        selected = {}
        for index in indices:
            key = tuple(np.floor(points[index] / spacing).astype(int))
            selected.setdefault(key, int(index))
        return np.array(list(selected.values()), dtype=int)

    @staticmethod
    def scale_values(values):
        values = np.asarray(values, dtype=float)
        minimum = float(values.min())
        spread = float(values.max() - minimum)
        if spread <= 1e-12:
            return np.zeros_like(values)
        return (values - minimum) / spread
