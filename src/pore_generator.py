"""
Module for generating various types of pores.
"""
import random
from typing import Any, Dict, List, Tuple

import cv2
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
        # Добавляем маску занятых областей для лучшего контроля размещения
        self.occupied_mask = None

    def generate_image(self) -> Tuple[np.ndarray, Dict[str, List]]:
        """Generates an image with all configured pore types."""
        image = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Инициализируем маску занятых областей
        self.occupied_mask = np.zeros((self.height, self.width), dtype=bool)
        
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

    def _create_irregular_pore(self, radius: int, irregularity: float = 0.3, 
                               spikiness: float = 0.2, num_vertices: int = None) -> np.ndarray:
        """Creates an irregular pore shape using random polygon generation."""
        if num_vertices is None:
            num_vertices = random.randint(8, 16)
        
        # Canvas size with extra padding
        canvas_size = int(radius * 3)
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        center = canvas_size // 2
        
        # Generate angular steps
        angle_steps = []
        lower = (2 * np.pi / num_vertices) - (2 * np.pi / num_vertices) * irregularity
        upper = (2 * np.pi / num_vertices) + (2 * np.pi / num_vertices) * irregularity
        angle_sum = 0
        
        for i in range(num_vertices):
            tmp = random.uniform(lower, upper)
            angle_steps.append(tmp)
            angle_sum += tmp
            
        # Normalize angles
        angle_cumsum = 0
        angle_steps = [a * 2 * np.pi / angle_sum for a in angle_steps]
        
        # Generate points
        points = []
        for i in range(num_vertices):
            angle_cumsum += angle_steps[i]
            
            # Add variation to radius
            r_i = radius * (1 + random.uniform(-spikiness, spikiness))
            r_i = max(radius * 0.5, min(radius * 1.5, r_i))  # Limit variation
            
            x = center + int(r_i * np.cos(angle_cumsum))
            y = center + int(r_i * np.sin(angle_cumsum))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(canvas, [points], 0)
        
        # Apply smoothing for more natural look
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        
        # Add small random perturbations to edges
        if random.random() > 0.3:
            canvas = self._add_edge_roughness(canvas)
        
        return canvas

    def _add_edge_roughness(self, pore_mask: np.ndarray) -> np.ndarray:
        """Adds micro-roughness to pore edges for more realistic appearance."""
        # Create small random kernel for edge perturbation
        kernel_size = random.choice([3, 5])
        kernel = np.random.rand(kernel_size, kernel_size)
        kernel = (kernel > 0.6).astype(np.uint8)
        
        # Randomly erode or dilate edges
        if random.random() > 0.5:
            # Add small protrusions
            dilated = cv2.dilate(255 - pore_mask, kernel, iterations=1)
            result = 255 - dilated
        else:
            # Add small indentations
            eroded = cv2.erode(255 - pore_mask, kernel, iterations=1)
            result = 255 - eroded
            
        return result

    def _create_realistic_pore(self, radius: int, pore_type: str = 'normal') -> np.ndarray:
        """Creates a realistic pore with natural irregular boundaries."""
        
        if pore_type == 'smooth':
            # Start with a slightly irregular circle
            irregularity = random.uniform(0.1, 0.25)
            spikiness = random.uniform(0.05, 0.15)
            return self._create_irregular_pore(radius, irregularity, spikiness, random.randint(12, 20))
            
        elif pore_type == 'rough':
            # More irregular shape
            irregularity = random.uniform(0.25, 0.4)
            spikiness = random.uniform(0.15, 0.3)
            pore = self._create_irregular_pore(radius, irregularity, spikiness, random.randint(8, 14))
            
            # Add Perlin noise distortion
            return self._apply_perlin_distortion(pore, radius, deformation=0.2)
            
        elif pore_type == 'defective':
            # Highly irregular shape with Perlin noise
            base_pore = self._create_irregular_pore(radius, 0.4, 0.3, random.randint(6, 12))
            return self._apply_perlin_distortion(base_pore, radius, deformation=0.35)
            
        else:  # 'normal'
            irregularity = random.uniform(0.15, 0.3)
            spikiness = random.uniform(0.1, 0.2)
            return self._create_irregular_pore(radius, irregularity, spikiness)

    def _apply_perlin_distortion(self, pore_mask: np.ndarray, radius: int, deformation: float) -> np.ndarray:
        """Applies Perlin noise distortion to a pore mask."""
        h, w = pore_mask.shape
        center = h // 2
        
        # Apply distortion
        distorted = np.ones_like(pore_mask) * 255
        coords = np.argwhere(pore_mask == 0)
        
        # Создаем временную маску для заполнения
        temp_mask = np.zeros_like(pore_mask, dtype=bool)

        noise_field = np.zeros((h, w))
        
        for y, x in coords:
            # Calculate offset based on noise
            offset = noise_field[y, x] * radius * 0.3
            angle = np.arctan2(y - center, x - center)
            
            new_x = int(x + offset * np.cos(angle))
            new_y = int(y + offset * np.sin(angle))
            
            if 0 <= new_x < w and 0 <= new_y < h:
                temp_mask[new_y, new_x] = True
        
        # Заполняем область используя морфологическое закрытие
        distorted[temp_mask] = 0
        
        # Важно: используем БОЛЬШИЙ kernel для заполнения дыр
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        distorted = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Apply slight Gaussian blur for smoother edges
        distorted = cv2.GaussianBlur(distorted, (3, 3), 0.5)
        _, distorted = cv2.threshold(distorted, 127, 255, cv2.THRESH_BINARY)
        
        return distorted

    def _can_place_pore(self, pore_canvas: np.ndarray, x: int, y: int, 
                       canvas_center: int, min_distance: int = 3) -> bool:
        """Проверяет, можно ли разместить пору без пересечений."""
        pore_height, pore_width = pore_canvas.shape
        
        # Рассчитываем границы размещения
        y_start_img = max(0, y - canvas_center)
        y_end_img = min(self.height, y - canvas_center + pore_height)
        x_start_img = max(0, x - canvas_center)
        x_end_img = min(self.width, x - canvas_center + pore_width)
        
        y_start_pore = max(0, canvas_center - y)
        y_end_pore = y_start_pore + (y_end_img - y_start_img)
        x_start_pore = max(0, canvas_center - x)
        x_end_pore = x_start_pore + (x_end_img - x_start_img)
        
        # Получаем область поры
        pore_region = pore_canvas[y_start_pore:y_end_pore, x_start_pore:x_end_pore]
        pore_mask = pore_region < 255
        
        # Проверяем пересечение с уже занятыми областями
        occupied_region = self.occupied_mask[y_start_img:y_end_img, x_start_img:x_end_img]
        
        # Расширяем маску поры для обеспечения минимального расстояния
        if min_distance > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (2*min_distance+1, 2*min_distance+1))
            expanded_pore_mask = cv2.dilate(pore_mask.astype(np.uint8), kernel) > 0
        else:
            expanded_pore_mask = pore_mask
        
        # Проверяем пересечение
        if np.any(occupied_region & expanded_pore_mask):
            return False
            
        return True

    def _place_on_image(self, image: np.ndarray, canvas: np.ndarray, 
                       canvas_center: int, x: int, y: int, 
                       update_occupied: bool = True) -> np.ndarray:
        """Улучшенное размещение без артефактов"""
        pore_height, pore_width = canvas.shape

        y_start_img = max(0, y - canvas_center)
        y_end_img = min(self.height, y - canvas_center + pore_height)
        x_start_img = max(0, x - canvas_center)
        x_end_img = min(self.width, x - canvas_center + pore_width)

        y_start_pore = max(0, canvas_center - y)
        y_end_pore = y_start_pore + (y_end_img - y_start_img)
        x_start_pore = max(0, canvas_center - x)
        x_end_pore = x_start_pore + (x_end_img - x_start_img)

        # Используем маску для более точного наложения
        pore_slice = canvas[y_start_pore:y_end_pore, x_start_pore:x_end_pore]
        mask = pore_slice < 255  # Только черные пиксели поры
        
        image[y_start_img:y_end_img, x_start_img:x_end_img][mask] = 0
        
        # Обновляем маску занятых областей
        if update_occupied and self.occupied_mask is not None:
            self.occupied_mask[y_start_img:y_end_img, x_start_img:x_end_img][mask] = True

        return image

    def _add_single_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds single, non-overlapping pores to the image."""
        settings = self.config['pore_settings']['single_pores']
        count = random.randint(*settings['count_range'])

        for _ in range(count):
            radius = np.random.normal(settings['radius_mean'], settings['radius_std'])
            radius = int(np.clip(radius, settings['min_radius'], settings['max_radius']))

            pore_type = random.choice(['smooth', 'normal', 'rough'])
            pore_canvas = self._create_realistic_pore(radius, pore_type)
            
            canvas_size = pore_canvas.shape[0]
            center = canvas_size // 2
            
            # Увеличиваем отступ от краев
            margin = center + 10

            for _ in range(200):  # Увеличиваем количество попыток
                x = random.randint(margin, self.width - margin)
                y = random.randint(margin, self.height - margin)

                # Используем улучшенную проверку размещения
                if self._can_place_pore(pore_canvas, x, y, center, min_distance=5):
                    self._place_on_image(image, pore_canvas, center, x, y)
                    pore_info['single'].append({
                        'x': x, 'y': y, 'radius': radius,
                        'type': pore_type
                    })
                    break
        return image

    def _add_weakly_overlapping_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds groups of weakly overlapping pores, separated by the watershed algorithm."""
        settings = self.config['pore_settings']['weakly_overlapping']
        group_count = random.randint(*settings['count_range'])

        for _ in range(group_count):
            # Ищем свободное место для группы
            group_placed = False
            for attempt in range(50):
                temp_image = np.ones((self.height, self.width), dtype=np.uint8) * 255
                temp_occupied = np.zeros((self.height, self.width), dtype=bool)
                pore_count = random.randint(2, 3)
                group_details = {'centers': [], 'radii': [], 'types': []}
                group_valid = True

                for i in range(pore_count):
                    radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 5, 25))
                    pore_type = random.choice(['smooth', 'normal', 'rough'])
                    
                    effective_radius = radius
                    margin = int(np.ceil(effective_radius * 2))

                    if i == 0:
                        x = random.randint(margin, self.width - margin)
                        y = random.randint(margin, self.height - margin)
                        
                        # Проверяем, свободна ли область для первой поры группы
                        test_canvas = self._create_realistic_pore(radius, pore_type)
                        if not self._can_place_pore(test_canvas, x, y, test_canvas.shape[0]//2, min_distance=10):
                            group_valid = False
                            break
                    else:
                        overlap_percent = random.uniform(*settings['overlap_percentage_range'])
                        prev_x, prev_y = group_details['centers'][-1]
                        prev_radius = group_details['radii'][-1]
                        
                        distance = (prev_radius + effective_radius) * (1 - overlap_percent)
                        direction_angle = random.uniform(0, 2 * np.pi)
                        
                        x = int(prev_x + distance * np.cos(direction_angle))
                        y = int(prev_y + distance * np.sin(direction_angle))
                    
                    x = max(margin, min(x, self.width - margin))
                    y = max(margin, min(y, self.height - margin))
                    
                    pore_canvas = self._create_realistic_pore(radius, pore_type)
                    canvas_size = pore_canvas.shape[0]
                    center = canvas_size // 2
                    
                    # Размещаем на временном изображении
                    self._place_on_image_temp(temp_image, pore_canvas, center, x, y, temp_occupied)

                    group_details['centers'].append((x, y))
                    group_details['radii'].append(radius)
                    group_details['types'].append(pore_type)

                if group_valid:
                    # Проверяем, не пересекается ли вся группа с существующими порами
                    group_mask = temp_image < 255
                    if not np.any(self.occupied_mask & group_mask):
                        separated_image = self._apply_watershed(temp_image)
                        image = np.minimum(image, separated_image)
                        # Обновляем маску занятых областей
                        self.occupied_mask |= (separated_image < 255)
                        pore_info['weakly_overlapping'].append(group_details)
                        group_placed = True
                        break

        return image

    def _place_on_image_temp(self, image: np.ndarray, canvas: np.ndarray, 
                            canvas_center: int, x: int, y: int, 
                            temp_occupied: np.ndarray) -> np.ndarray:
        """Временное размещение для проверки."""
        pore_height, pore_width = canvas.shape

        y_start_img = max(0, y - canvas_center)
        y_end_img = min(self.height, y - canvas_center + pore_height)
        x_start_img = max(0, x - canvas_center)
        x_end_img = min(self.width, x - canvas_center + pore_width)

        y_start_pore = max(0, canvas_center - y)
        y_end_pore = y_start_pore + (y_end_img - y_start_img)
        x_start_pore = max(0, canvas_center - x)
        x_end_pore = x_start_pore + (x_end_img - x_start_img)

        pore_slice = canvas[y_start_pore:y_end_pore, x_start_pore:x_end_pore]
        mask = pore_slice < 255
        
        image[y_start_img:y_end_img, x_start_img:x_end_img][mask] = 0
        temp_occupied[y_start_img:y_end_img, x_start_img:x_end_img][mask] = True

        return image

    def _add_strongly_overlapping_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds groups of strongly overlapping (merged) pores."""
        settings = self.config['pore_settings']['strongly_overlapping']
        group_count = random.randint(*settings['count_range'])
        
        for _ in range(group_count):
            # Ищем свободное место для группы
            for attempt in range(50):
                pore_count = random.randint(2, 4)
                group_details = {'centers': [], 'radii': [], 'types': []}
                group_valid = True

                # Создаем временное изображение для проверки
                temp_image = np.ones((self.height, self.width), dtype=np.uint8) * 255

                for i in range(pore_count):
                    radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 5, 20))
                    pore_type = random.choice(['normal', 'rough'])

                    effective_radius = radius
                    margin = int(np.ceil(effective_radius * 2))

                    if i == 0:
                        x = random.randint(margin, self.width - margin)
                        y = random.randint(margin, self.height - margin)
                        
                        # Проверяем область для первой поры
                        test_canvas = self._create_realistic_pore(radius, pore_type)
                        if not self._can_place_pore(test_canvas, x, y, test_canvas.shape[0]//2, min_distance=10):
                            group_valid = False
                            break
                    else:
                        overlap_percent = random.uniform(*settings['overlap_percentage_range'])
                        prev_x, prev_y = group_details['centers'][-1]
                        prev_radius = group_details['radii'][-1]

                        distance = (prev_radius + effective_radius) * (1 - overlap_percent)
                        direction_angle = random.uniform(0, 2 * np.pi)

                        x = int(prev_x + distance * np.cos(direction_angle))
                        y = int(prev_y + distance * np.sin(direction_angle))

                    x = max(margin, min(x, self.width - margin))
                    y = max(margin, min(y, self.height - margin))

                    pore_canvas = self._create_realistic_pore(radius, pore_type)
                    canvas_size = pore_canvas.shape[0]
                    center = canvas_size // 2
                    
                    # Размещаем на временном изображении
                    self._place_on_image(temp_image, pore_canvas, center, x, y, update_occupied=False)
                    
                    group_details['centers'].append((x, y))
                    group_details['radii'].append(radius)
                    group_details['types'].append(pore_type)

                if group_valid:
                    # Проверяем всю группу
                    group_mask = temp_image < 255
                    if not np.any(self.occupied_mask & group_mask):
                        # Размещаем на основном изображении
                        image = np.minimum(image, temp_image)
                        self.occupied_mask |= group_mask
                        pore_info['strongly_overlapping'].append(group_details)
                        break

        return image

    def _add_defective_pores(self, image: np.ndarray, pore_info: Dict[str, List]) -> np.ndarray:
        """Adds irregularly shaped defective pores."""
        settings = self.config['pore_settings']['defective_pores']
        count = random.randint(*settings['count_range'])

        for _ in range(count):
            radius = int(np.clip(np.random.normal(settings['radius_mean'], settings['radius_std']), 10, 30))
            
            # Always use defective type for these pores
            defective_pore_canvas = self._create_realistic_pore(radius, 'defective')
            
            canvas_size = defective_pore_canvas.shape[0]
            center = canvas_size // 2
            margin = center + 10

            for _ in range(200):  # Много попыток найти место
                x = random.randint(margin, self.width - margin)
                y = random.randint(margin, self.height - margin)

                if self._can_place_pore(defective_pore_canvas, x, y, center, min_distance=8):
                    self._place_on_image(image, defective_pore_canvas, center, x, y)
                    pore_info['defective'].append({
                        'x': x, 'y': y, 'radius': radius,
                        'type': 'defective'
                    })
                    break

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
