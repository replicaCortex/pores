"""
Module for generating various types of pores.
"""
import random
from typing import Any, Dict, List, Tuple

import cv2
import noise
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class PoreGenerator:
    """Generates images with various types of simulated pores."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config['image_settings']['width']
        self.height = config['image_settings']['height']

    def generate_image(self) -> Tuple[np.ndarray, Dict[str, List]]:
        """Generates an image with all configured pore types."""
        image = np.ones((self.height, self.width), dtype=np.uint8) * 255
        pore_info = {
            'single': [],
            'weakly_overlapping': [],
            'strongly_overlapping': [],
            'defective': []
        }

        image = self._add_single_pores(image, pore_info)
        image = self._add_weakly_overlapping_pores(image, pore_info)
        image = self._add_strongly_overlapping_pores(image, pore_info)
        image = self._add_defective_pores(image, pore_info)

        return image, pore_info

    def _create_elliptical_pore(self, radius: int, stretch_factor: float, angle: float) -> Tuple[np.ndarray, int]:
        """Creates an elliptical pore on its own canvas."""
        canvas_size = int(radius * 2 * stretch_factor * 1.5)
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        center = canvas_size // 2

        axes = (int(radius * stretch_factor), radius)
        cv2.ellipse(canvas, (center, center), axes, angle, 0, 360, 0, -1)

        return canvas, center

    def _place_on_image(self, image: np.ndarray, canvas: np.ndarray, canvas_center: int, x: int, y: int) -> np.ndarray:
        """Places a pore from its canvas onto the main image, handling boundaries."""
        pore_height, pore_width = canvas.shape

        y_start_img = max(0, y - canvas_center)
        y_end_img = min(self.height, y - canvas_center + pore_height)
        x_start_img = max(0, x - canvas_center)
        x_end_img = min(self.width, x - canvas_center + pore_width)

        y_start_pore = max(0, canvas_center - y)
        y_end_pore = y_start_pore + (y_end_img - y_start_img)
        x_start_pore = max(0, canvas_center - x)
        x_end_pore = x_start_pore + (x_end_img - x_start_img)

        img_slice = image[y_start_img:y_end_img, x_start_img:x_end_img]
        pore_slice = canvas[y_start_pore:y_end_pore, x_start_pore:x_end_pore]
        
        image[y_start_img:y_end_img, x_start_img:x_end_img] = np.minimum(img_slice, pore_slice)

        return image

    def _add_single_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds single, non-overlapping pores to the image."""
        settings = self.config['pore_settings']['single_pores']
        count = random.randint(*settings['count_range'])

        for _ in range(count):
            radius = np.random.normal(settings['radius_mean'], settings['radius_std'])
            radius = int(np.clip(radius, settings['min_radius'], settings['max_radius']))

            stretch_factor = 1.0
            angle = 0.0
            if settings['stretch_enabled']:
                stretch_factor = random.uniform(*settings['stretch_factor_range'])
                if settings['rotation_enabled']:
                    angle = random.uniform(0, 180)

            effective_radius = int(radius * stretch_factor)
            if stretch_factor > 1.0:
                pore_canvas, center = self._create_elliptical_pore(radius, stretch_factor, angle)
            else:
                canvas_size = radius * 2 + 1
                pore_canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
                cv2.circle(pore_canvas, (radius, radius), radius, 0, -1)
                center = radius

            placed = False
            for _ in range(100):  # Max attempts
                x = random.randint(effective_radius, self.width - effective_radius)
                y = random.randint(effective_radius, self.height - effective_radius)

                if self._is_area_free(image, x, y, effective_radius):
                    self._place_on_image(image, pore_canvas, center, x, y)
                    pore_info['single'].append({
                        'x': x, 'y': y, 'radius': radius,
                        'stretch_factor': stretch_factor, 'angle': angle
                    })
                    placed = True
                    break
        return image

    def _add_weakly_overlapping_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds groups of weakly overlapping pores, separated by the watershed algorithm."""
        settings = self.config['pore_settings']['weakly_overlapping']
        group_count = random.randint(*settings['count_range'])

        for _ in range(group_count):
            temp_image = np.ones((self.height, self.width), dtype=np.uint8) * 255
            pore_count = random.randint(2, 3)
            group_details = {'centers': [], 'radii': [], 'stretch_factors': [], 'angles': []}

            for i in range(pore_count):
                radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 5, 25))
                
                stretch_factor = 1.0
                angle = 0.0
                if settings['stretch_enabled']:
                    stretch_factor = random.uniform(*settings['stretch_factor_range'])
                    if settings['rotation_enabled']:
                        angle = random.uniform(0, 180)
                
                effective_radius = radius * stretch_factor
                margin = int(np.ceil(effective_radius))

                if i == 0:
                    x = random.randint(margin, self.width - margin)
                    y = random.randint(margin, self.height - margin)
                else:
                    overlap_percent = random.uniform(*settings['overlap_percentage_range'])
                    prev_x, prev_y = group_details['centers'][-1]
                    prev_radius = group_details['radii'][-1]
                    prev_stretch = group_details['stretch_factors'][-1]
                    
                    effective_prev_radius = prev_radius * prev_stretch
                    distance = (effective_prev_radius + effective_radius) * (1 - overlap_percent)
                    direction_angle = random.uniform(0, 2 * np.pi)
                    
                    x = int(prev_x + distance * np.cos(direction_angle))
                    y = int(prev_y + distance * np.sin(direction_angle))
                
                x = max(margin, min(x, self.width - margin))
                y = max(margin, min(y, self.height - margin))
                
                if stretch_factor > 1.0:
                    pore_canvas, center = self._create_elliptical_pore(radius, stretch_factor, angle)
                    self._place_on_image(temp_image, pore_canvas, center, x, y)
                else:
                    cv2.circle(temp_image, (x, y), radius, 0, -1)

                group_details['centers'].append((x, y))
                group_details['radii'].append(radius)
                group_details['stretch_factors'].append(stretch_factor)
                group_details['angles'].append(angle)

            separated_image = self._apply_watershed(temp_image)
            image = np.minimum(image, separated_image)
            pore_info['weakly_overlapping'].append(group_details)

        return image

    def _add_strongly_overlapping_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds groups of strongly overlapping (merged) pores."""
        settings = self.config['pore_settings']['strongly_overlapping']
        group_count = random.randint(*settings['count_range'])
        
        for _ in range(group_count):
            pore_count = random.randint(2, 4)
            group_details = {'centers': [], 'radii': [], 'stretch_factors': [], 'angles': []}

            for i in range(pore_count):
                radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 5, 20))

                stretch_factor = 1.0
                angle = 0.0
                if settings.get('stretch_enabled', False):
                    stretch_factor = random.uniform(*settings['stretch_factor_range'])
                    if settings.get('rotation_enabled', False):
                        angle = random.uniform(0, 180)

                effective_radius = radius * stretch_factor
                margin = int(np.ceil(effective_radius))

                if i == 0:
                    x = random.randint(margin, self.width - margin)
                    y = random.randint(margin, self.height - margin)
                else:
                    overlap_percent = random.uniform(*settings['overlap_percentage_range'])
                    prev_x, prev_y = group_details['centers'][-1]
                    prev_radius = group_details['radii'][-1]
                    prev_stretch = group_details['stretch_factors'][-1]

                    effective_prev_radius = prev_radius * prev_stretch
                    distance = (effective_prev_radius + effective_radius) * (1 - overlap_percent)
                    direction_angle = random.uniform(0, 2 * np.pi)

                    x = int(prev_x + distance * np.cos(direction_angle))
                    y = int(prev_y + distance * np.sin(direction_angle))

                x = max(margin, min(x, self.width - margin))
                y = max(margin, min(y, self.height - margin))

                if stretch_factor > 1.0:
                    pore_canvas, center = self._create_elliptical_pore(radius, stretch_factor, angle)
                    self._place_on_image(image, pore_canvas, center, x, y)
                else:
                    cv2.circle(image, (x, y), radius, 0, -1)
                
                group_details['centers'].append((x, y))
                group_details['radii'].append(radius)
                group_details['stretch_factors'].append(stretch_factor)
                group_details['angles'].append(angle)

            pore_info['strongly_overlapping'].append(group_details)

        return image

    def _add_defective_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds irregularly shaped defective pores."""
        settings = self.config['pore_settings']['defective_pores']
        count = random.randint(*settings['count_range'])

        for _ in range(count):
            radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 10, 30))
            
            stretch_factor = 1.0
            angle = 0.0
            if settings['stretch_enabled']:
                stretch_factor = random.uniform(*settings['stretch_factor_range'])
                if settings['rotation_enabled']:
                    angle = random.uniform(0, 180)

            x = random.randint(radius * 2, self.width - radius * 2)
            y = random.randint(radius * 2, self.height - radius * 2)

            defective_pore_canvas = self._create_defective_pore(
                radius, settings['deformation_factor'], stretch_factor, angle
            )
            
            pore_h, pore_w = defective_pore_canvas.shape
            center = max(pore_h, pore_w) // 2
            self._place_on_image(image, defective_pore_canvas, center, x, y)

            pore_info['defective'].append({
                'x': x, 'y': y, 'radius': radius,
                'stretch_factor': stretch_factor, 'angle': angle
            })

        return image

    def _is_area_free(self, image: np.ndarray, x: int, y: int, radius: int, padding: int = 5) -> bool:
        """Checks if a square area around a point is free (white)."""
        y_start = max(0, y - radius - padding)
        y_end = min(self.height, y + radius + padding)
        x_start = max(0, x - radius - padding)
        x_end = min(self.width, x + radius + padding)

        return np.all(image[y_start:y_end, x_start:x_end] == 255)

    def _apply_watershed(self, image: np.ndarray) -> np.ndarray:
        """Applies the watershed algorithm to separate touching objects."""
        inverted = cv2.bitwise_not(image)
        distance = ndimage.distance_transform_edt(inverted)
        
        coords = peak_local_max(distance, min_distance=3)
        local_maxima = np.zeros_like(distance, dtype=bool)
        local_maxima[tuple(coords.T)] = True
        
        markers = ndimage.label(local_maxima)[0]
        labels = watershed(-distance, markers, mask=inverted)
        
        separated_image = np.ones_like(image) * 255
        for label in np.unique(labels):
            if label == 0:
                continue
            separated_image[labels == label] = 0
            
        edges = cv2.Canny(labels.astype(np.uint8), 1, 2)
        separated_image[edges > 0] = 255
        
        return separated_image

    def _create_defective_pore(self, radius: int, deformation: float, stretch: float, angle: float) -> np.ndarray:
        """Creates a single deformed pore using Perlin noise."""
        canvas_size = int(radius * 2 * stretch * 1.5)
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        center = canvas_size // 2

        axes = (int(radius * stretch), radius)
        temp_ellipse = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.ellipse(temp_ellipse, (center, center), axes, angle, 0, 360, 255, -1)

        scale, octaves, persistence = 0.1, 2, 0.5
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for i in range(canvas_size):
            for j in range(canvas_size):
                if temp_ellipse[i, j] > 0:
                    noise_val = noise.pnoise2(i * scale, j * scale, octaves=octaves, persistence=persistence)
                    
                    dx, dy = j - center, i - center
                    x_rot = dx * cos_a + dy * sin_a
                    y_rot = -dx * sin_a + dy * cos_a
                    
                    safe_radius = max(radius, 1)
                    safe_stretch_radius = max(radius * stretch, 1)

                    norm_dist = np.sqrt((x_rot / safe_stretch_radius)**2 + (y_rot / safe_radius)**2)
                    
                    modified_boundary = 1.0 + noise_val * deformation
                    if norm_dist <= modified_boundary:
                        canvas[i, j] = 0

        coords = np.argwhere(canvas == 0)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cropped = canvas[y_min:y_max+1, x_min:x_max+1]
            return cropped
        
        return np.ones((1,1), dtype=np.uint8) * 255 # Return a blank pixel if empty
