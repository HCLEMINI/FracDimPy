#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Pattern Generators
===========================

This module implements various fractal pattern generation algorithms:

- Cantor set
- Sierpinski triangle and carpet
- DLA (Diffusion-Limited Aggregation)
- Menger sponge and related patterns
"""

import numpy as np

# type: ignore
from typing import Tuple, Optional

from ..utils.image_drawing import bresenham_line, normalize_to_pixel_coords


def _koch_segment(p1, p2, level):
    """Recursively generate Koch curve segment.

    Parameters
    ----------
    p1 : array-like
        Start point (x, y).
    p2 : array-like
        End point (x, y).
    level : int
        Recursion depth.

    Returns
    -------
    points : list
        List of (x, y) point arrays forming the Koch segment.
    """
    if level == 0:
        return [p1, p2]

    # Calculate direction vector
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Calculate trisection points
    p1_3 = np.array([p1[0] + dx / 3, p1[1] + dy / 3])
    p2_3 = np.array([p1[0] + 2 * dx / 3, p1[1] + 2 * dy / 3])

    # Calculate peak point (60° outward)
    # Forms an equilateral triangle
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2

    # Rotate vector to peak by 90 degrees
    vx = p1_3[0] - cx
    vy = p1_3[1] - cy

    # Perpendicular vector (rotated 90 degrees)
    peak = np.array([cx - vy * np.sqrt(3), cy + vx * np.sqrt(3)])

    # Recursively generate four segments
    points = []
    points.extend(_koch_segment(p1, p1_3, level - 1)[:-1])
    points.extend(_koch_segment(p1_3, peak, level - 1)[:-1])
    points.extend(_koch_segment(peak, p2_3, level - 1)[:-1])
    points.extend(_koch_segment(p2_3, p2, level - 1))

    return points


def generate_cantor_set(level: int = 5, length: int = 3**5) -> np.ndarray:
    """
    Generate a Cantor set fractal pattern.

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 5
    length : int, optional
        Array length, should be 3^level, default is 243

    Returns
    -------
    cantor : np.ndarray
        Cantor set as 1D array with 1s and 0s

    Examples
    --------
    >>> from fracDimPy import generate_cantor_set
    >>> cantor = generate_cantor_set(level=5)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow([cantor], cmap='binary', aspect='auto')
    >>> plt.show()

    Notes
    -----
    The Cantor set has fractal dimension log(2)/log(3) ≈ 0.631
    """
    arr = np.ones(length)

    for i in range(level):
        segment_length = length // (3 ** (i + 1))
        for j in range(3**i):
            start = j * (3 * segment_length) + segment_length
            end = start + segment_length
            arr[start:end] = 0

    return arr


def generate_sierpinski(level: int = 5, size: int = 512) -> np.ndarray:
    """
    Generate a Sierpinski triangle fractal.

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 5
    size : int, optional
        Image size (pixels), default is 512

    Returns
    -------
    triangle : np.ndarray
        2D array representing the Sierpinski triangle

    Examples
    --------
    >>> from fracDimPy import generate_sierpinski
    >>> triangle = generate_sierpinski(level=5)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(triangle, cmap='binary')
    >>> plt.show()

    Notes
    -----
    The Sierpinski triangle has fractal dimension log(3)/log(2) ≈ 1.585
    It is constructed by recursively removing middle triangles from an
    equilateral triangle, with 3 remaining triangles at each level.
    """
    img = np.zeros((size, size), dtype=np.uint8)

    def fill_triangle(x1, y1, x2, y2, x3, y3):
        """Fill a solid triangle"""
        # Use scanline algorithm to fill the triangle
        # Find bounding box
        min_y = max(0, int(min(y1, y2, y3)))
        max_y = min(size - 1, int(max(y1, y2, y3)))

        for y in range(min_y, max_y + 1):
            x_intersects = []

            # Find intersections with each edge
            edges = [(x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x1, y1)]
            for ex1, ey1, ex2, ey2 in edges:
                if ey1 != ey2:  # Skip horizontal edges
                    if min(ey1, ey2) <= y <= max(ey1, ey2):
                        x = ex1 + (y - ey1) * (ex2 - ex1) / (ey2 - ey1)
                        x_intersects.append(x)

            if len(x_intersects) >= 2:
                x_intersects.sort()
                x_min = max(0, int(x_intersects[0]))
                x_max = min(size - 1, int(x_intersects[-1]))
                img[y, x_min : x_max + 1] = 1

    def remove_middle_triangle(x1, y1, x2, y2, x3, y3):
        """Remove the middle triangle"""
        # Calculate midpoints of each edge
        mx12 = (x1 + x2) / 2
        my12 = (y1 + y2) / 2
        mx23 = (x2 + x3) / 2
        my23 = (y2 + y3) / 2
        mx31 = (x3 + x1) / 2
        my31 = (y3 + y1) / 2

        # Set middle triangle to 0
        min_y = max(0, int(min(my12, my23, my31)))
        max_y = min(size - 1, int(max(my12, my23, my31)))

        for y in range(min_y, max_y + 1):
            x_intersects = []
            edges = [(mx12, my12, mx23, my23), (mx23, my23, mx31, my31), (mx31, my31, mx12, my12)]
            for ex1, ey1, ex2, ey2 in edges:
                if ey1 != ey2:
                    if min(ey1, ey2) <= y <= max(ey1, ey2):
                        x = ex1 + (y - ey1) * (ex2 - ex1) / (ey2 - ey1)
                        x_intersects.append(x)

            if len(x_intersects) >= 2:
                x_intersects.sort()
                x_min = max(0, int(x_intersects[0]))
                x_max = min(size - 1, int(x_intersects[-1]))
                img[y, x_min : x_max + 1] = 0

    def sierpinski_recursive(x1, y1, x2, y2, x3, y3, depth):
        """Recursively generate Sierpinski triangle"""
        if depth == 0:
            return

        # Remove middle triangle
        remove_middle_triangle(x1, y1, x2, y2, x3, y3)

        if depth > 1:
            # Calculate midpoints
            mx12 = (x1 + x2) / 2
            my12 = (y1 + y2) / 2
            mx23 = (x2 + x3) / 2
            my23 = (y2 + y3) / 2
            mx31 = (x3 + x1) / 2
            my31 = (y3 + y1) / 2

            # Recursively process three sub-triangles
            sierpinski_recursive(x1, y1, mx12, my12, mx31, my31, depth - 1)  # Top
            sierpinski_recursive(mx12, my12, x2, y2, mx23, my23, depth - 1)  # Bottom-left
            sierpinski_recursive(mx31, my31, mx23, my23, x3, y3, depth - 1)  # Bottom-right

    # Define initial equilateral triangle
    height = int(size * 0.866)  # sqrt(3)/2 ≈ 0.866
    x1, y1 = size // 2, 10  # Top vertex
    x2, y2 = 10, height  # Bottom-left vertex
    x3, y3 = size - 10, height  # Bottom-right vertex

    # Fill initial triangle
    fill_triangle(x1, y1, x2, y2, x3, y3)

    # Apply Sierpinski recursion
    sierpinski_recursive(x1, y1, x2, y2, x3, y3, level)

    return img


def generate_sierpinski_carpet(level: int = 5, size: int = 243) -> np.ndarray:
    """
    Generate a Sierpinski carpet fractal (2D analog of Menger sponge).

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 5
    size : int, optional
        Should be 3^level, default is 243 (3^5)

    Returns
    -------
    carpet : np.ndarray
        2D array representing the Sierpinski carpet

    Examples
    --------
    >>> from fracDimPy import generate_sierpinski_carpet
    >>> carpet = generate_sierpinski_carpet(level=5)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(carpet, cmap='binary')
    >>> plt.show()

    Notes
    -----
    The Sierpinski carpet is the 2D analog of the Menger sponge.
    It has fractal dimension log(8)/log(3) ≈ 1.893
    At each iteration, a 3x3 grid is created and the central square
    is removed, leaving 8 sub-squares.
    """
    carpet = np.ones((size, size), dtype=bool)

    def remove_centers(arr, depth):
        """Recursively remove central squares"""
        if depth == 0:
            return

        s = arr.shape[0]
        step = s // 3

        # Remove center square
        arr[step : 2 * step, step : 2 * step] = False

        # Recursively process 8 remaining squares
        if depth > 1:
            for i in range(3):
                for j in range(3):
                    # Skip center square
                    if i == 1 and j == 1:
                        continue
                    # Process sub-square
                    sub = arr[i * step : (i + 1) * step, j * step : (j + 1) * step]
                    remove_centers(sub, depth - 1)

    remove_centers(carpet, level)

    return carpet.astype(int)


def generate_vicsek_fractal(level: int = 5, size: int = 243) -> np.ndarray:
    """
    Generate a Vicsek fractal pattern.

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 5
    size : int, optional
        Should be 3^level, default is 243 (3^5)

    Returns
    -------
    vicsek : np.ndarray
        2D array representing the Vicsek fractal

    Examples
    --------
    >>> from fracDimPy import generate_vicsek_fractal
    >>> vicsek = generate_vicsek_fractal(level=5)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(vicsek, cmap='binary')
    >>> plt.show()

    Notes
    -----
    The Vicsek fractal has dimension log(5)/log(3) ≈ 1.465
    It is created by recursively keeping a plus sign (+) pattern:
    3 vertical cells plus 1 horizontal cell plus 1 center cell,
    removing the 4 corner cells.
    """
    fractal = np.ones((size, size), dtype=bool)

    def remove_corners(arr, depth):
        """Recursively remove corner squares"""
        if depth == 0:
            return

        s = arr.shape[0]
        step = s // 3

        # Remove four corner cells (keep plus sign pattern)
        # Top-center
        arr[0:step, step : 2 * step] = False
        # Bottom-center
        arr[2 * step : 3 * step, step : 2 * step] = False
        # Left-middle
        arr[step : 2 * step, 0:step] = False
        # Right-middle
        arr[step : 2 * step, 2 * step : 3 * step] = False

        # Recursively process 5 remaining cells (plus pattern + 4 corners)
        if depth > 1:
            positions = [
                (0, 0),  # Top-left corner
                (0, 2),  # Top-right corner
                (1, 1),  # Center
                (2, 0),  # Bottom-left corner
                (2, 2),  # Bottom-right corner
            ]

            for i, j in positions:
                sub = arr[i * step : (i + 1) * step, j * step : (j + 1) * step]
                remove_corners(sub, depth - 1)

    remove_corners(fractal, level)

    return fractal.astype(int)


def generate_koch_curve(level: int = 4, size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Koch curve fractal.

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 4
    size : int, optional
        Image size in pixels, default is 512

    Returns
    -------
    points : np.ndarray
        Koch curve points as (N, 2) array
    image : np.ndarray
        2D image of the Koch curve

    Examples
    --------
    >>> from fracDimPy import generate_koch_curve
    >>> points, image = generate_koch_curve(level=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(points[:, 0], points[:, 1])
    >>> plt.show()

    Notes
    -----
    The Koch curve has fractal dimension log(4)/log(3) ≈ 1.2619
    """

    # Define initial line segment
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])

    # Generate Koch curve
    points = _koch_segment(start, end, level)
    points = np.array(points)

    # Create image
    image = np.zeros((size, size), dtype=np.uint8)

    # Normalize coordinates to pixel space
    x_img, y_img = normalize_to_pixel_coords(points, size, margin=50)

    # Draw lines using Bresenham's algorithm
    for i in range(len(x_img) - 1):
        bresenham_line(image, x_img[i], y_img[i], x_img[i + 1], y_img[i + 1], increment=0)

    return points, image


def generate_koch_snowflake(level: int = 4, size: int = 512) -> np.ndarray:
    """
    Generate a Koch snowflake fractal.

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 4
    size : int, optional
        Image size in pixels, default is 512

    Returns
    -------
    snowflake : np.ndarray
        2D image of the Koch snowflake

    Examples
    --------
    >>> from fracDimPy import generate_koch_snowflake
    >>> snowflake = generate_koch_snowflake(level=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(snowflake, cmap='binary')
    >>> plt.show()

    Notes
    -----
    The Koch snowflake is formed by applying the Koch curve to three sides
    of an equilateral triangle.
    It has the same fractal dimension as the Koch curve: log(4)/log(3) ≈ 1.2619
    """

    # Define initial equilateral triangle
    height = np.sqrt(3) / 2
    triangle = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, height])]

    # Apply Koch curve to each side
    all_points = []
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i + 1) % 3]
        segment_points = _koch_segment(p1, p2, level)[:-1]  # Exclude last point to avoid duplication
        all_points.extend(segment_points)

    points = np.array(all_points)

    # Create image
    image = np.zeros((size, size), dtype=np.uint8)

    # Normalize coordinates to pixel space
    x_img, y_img = normalize_to_pixel_coords(points, size, margin=50)

    # Draw lines
    for i in range(len(x_img)):
        x1 = x_img[(i + 1) % len(x_img)]
        y1 = y_img[(i + 1) % len(y_img)]
        bresenham_line(image, x_img[i], y_img[i], x1, y1, increment=0)

    return image


def generate_brownian_motion(
    steps: int = 10000,
    size: int = 512,
    step_size: float = 1.0,
    num_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random walk (2D Brownian motion) paths.

    Parameters
    ----------
    steps : int, optional
        Number of steps, default is 10000
    size : int, optional
        Image size in pixels, default is 512
    step_size : float, optional
        Step length, default is 1.0
    num_paths : int, optional
        Number of paths to generate, default is 1

    Returns
    -------
    paths : np.ndarray
        Path coordinates, shape (num_paths, steps, 2)
    image : np.ndarray
        2D image visualizing the paths

    Examples
    --------
    >>> from fracDimPy import generate_brownian_motion
    >>> paths, image = generate_brownian_motion(steps=5000)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(paths[0, :, 0], paths[0, :, 1])
    >>> plt.show()

    Notes
    -----
    2D Brownian motion (random walk) has fractal dimension 2.
    """
    # Generate multiple paths
    rng = np.random.default_rng(seed)
    all_paths = []

    for _ in range(num_paths):
        # Generate random angles
        angles = rng.uniform(0, 2 * np.pi, steps)

        # Compute displacements
        dx = step_size * np.cos(angles)
        dy = step_size * np.sin(angles)

        # Cumulative sum to get positions
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        # Add starting point
        x = np.concatenate([[0], x])
        y = np.concatenate([[0], y])

        path = np.column_stack([x, y])
        all_paths.append(path)

    paths = np.array(all_paths)

    # Create image
    image = np.zeros((size, size), dtype=np.uint8)

    # Normalize all paths together using global bounds
    all_points = paths.reshape(-1, 2)
    x_all, y_all = normalize_to_pixel_coords(all_points, size, margin=50)

    idx = 0
    for path in paths:
        n = len(path)
        x_img = x_all[idx : idx + n]
        y_img = y_all[idx : idx + n]
        idx += n

        # Draw path
        for i in range(len(x_img) - 1):
            bresenham_line(image, x_img[i], y_img[i], x_img[i + 1], y_img[i + 1], increment=5)

    return paths, image


def generate_levy_flight(
    steps: int = 5000,
    size: int = 512,
    alpha: float = 1.5,
    num_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lévy flight random walk paths.

    Parameters
    ----------
    steps : int, optional
        Number of steps, default is 5000
    size : int, optional
        Image size in pixels, default is 512
    alpha : float, optional
        Lévy exponent in range (0, 2], default is 1.5
        alpha=2 corresponds to Brownian motion
    num_paths : int, optional
        Number of paths to generate, default is 1

    Returns
    -------
    paths : np.ndarray
        Lévy flight path coordinates, shape (num_paths, steps, 2)
    image : np.ndarray
        2D image visualizing the Lévy flight

    Examples
    --------
    >>> from fracDimPy import generate_levy_flight
    >>> paths, image = generate_levy_flight(steps=3000, alpha=1.5)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(paths[0, :, 0], paths[0, :, 1])
    >>> plt.show()

    Notes
    -----
    Lévy flights are random walks with step lengths drawn from a
    heavy-tailed distribution, exhibiting occasional very long jumps.
    """
    if not (0 < alpha <= 2):
        raise ValueError("Lévy exponent alpha must be in range (0, 2]")

    rng = np.random.default_rng(seed)
    all_paths = []

    for _ in range(num_paths):
        # Generate random angles
        angles = rng.uniform(0, 2 * np.pi, steps)

        # Generate Lévy-distributed step lengths
        # Using Mantegna's algorithm
        import math

        sigma_u = (
            math.gamma(1 + alpha)
            * np.sin(np.pi * alpha / 2)
            / (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
        ) ** (1 / alpha)

        u = rng.normal(0, sigma_u, steps)
        v = rng.normal(0, 1, steps)

        step_lengths = u / (np.abs(v) ** (1 / alpha))

        # Clip extreme values for visualization
        step_lengths = np.clip(step_lengths, -100, 100)

        # Compute displacements
        dx = step_lengths * np.cos(angles)
        dy = step_lengths * np.sin(angles)

        # Cumulative sum
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        # Add starting point
        x = np.concatenate([[0], x])
        y = np.concatenate([[0], y])

        path = np.column_stack([x, y])
        all_paths.append(path)

    paths = np.array(all_paths)

    # Create image
    image = np.zeros((size, size), dtype=np.uint8)

    # Normalize all paths together using global bounds
    all_points = paths.reshape(-1, 2)
    x_all, y_all = normalize_to_pixel_coords(all_points, size, margin=50)

    idx = 0
    for path in paths:
        n = len(path)
        x_img = x_all[idx : idx + n]
        y_img = y_all[idx : idx + n]
        idx += n

        for i in range(len(x_img) - 1):
            bresenham_line(image, x_img[i], y_img[i], x_img[i + 1], y_img[i + 1], increment=5)

    return paths, image


def generate_self_avoiding_walk(
    steps: int = 5000,
    size: int = 512,
    num_attempts: int = 10,
    max_retries: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate self-avoiding random walk paths.

    Parameters
    ----------
    steps : int, optional
        Number of steps, default is 5000
    size : int, optional
        Image size in pixels, default is 512
    num_attempts : int, optional
        Number of walks to generate, default is 10
    max_retries : int, optional
        Maximum retries per walk, default is 1000

    Returns
    -------
    paths : list of np.ndarray
        List of successful walk paths
    image : np.ndarray
        2D image visualizing the walks

    Examples
    --------
    >>> from fracDimPy import generate_self_avoiding_walk
    >>> paths, image = generate_self_avoiding_walk(steps=3000)
    >>> import matplotlib.pyplot as plt
    >>> for path in paths:
    >>>     plt.plot(path[:, 0], path[:, 1], alpha=0.5)
    >>> plt.show()

    Notes
    -----
    Self-avoiding walks (SAW) cannot intersect themselves.

    Properties:
    - Mean-square displacement: R² ~ N^(2ν) where ν is the Flory exponent
    - 2D: ν ≈ 3/4 (conjectured)
    - 3D: ν ≈ 0.588
    - Fractal dimension D = 1/ν ≈ 4/3 in 2D
    - NP-hard to generate long SAWs

    Implementation notes:
    - Uses backtracking when stuck
    - May fail to generate full length for large step counts
    - Generates up to 4 directions at each step
    """
    rng = np.random.default_rng(seed)
    successful_paths = []

    # Define 4 cardinal directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for attempt in range(num_attempts):
        for retry in range(max_retries):
            # Start new walk
            path = [(0, 0)]
            visited = {(0, 0)}

            success = True

            for step in range(steps):
                current = path[-1]

                # Find available directions
                available_dirs = []
                for dx, dy in directions:
                    next_pos = (current[0] + dx, current[1] + dy)
                    if next_pos not in visited:
                        available_dirs.append(next_pos)

                if not available_dirs:
                    # Stuck - retry
                    success = False
                    break

                # Choose random available direction
                next_pos = available_dirs[rng.integers(len(available_dirs))]
                path.append(next_pos)
                visited.add(next_pos)

            if success:
                successful_paths.append(np.array(path))
                break

        if len(successful_paths) >= num_attempts:
            break

    if not successful_paths:
        raise RuntimeError(
            f"Failed to generate self-avoiding walks.\n"
            f"Try reducing steps={steps} or increasing max_retries={max_retries}"
        )

    # Create image
    image = np.zeros((size, size), dtype=np.uint8)

    # Normalize all paths together using global bounds
    all_points = np.vstack(successful_paths)
    x_all, y_all = normalize_to_pixel_coords(all_points, size, margin=50)

    idx = 0
    for path in successful_paths:  # type: ignore[assignment]
        n = len(path)
        x_img = x_all[idx : idx + n]
        y_img = y_all[idx : idx + n]
        idx += n

        # Draw path
        for i in range(len(x_img) - 1):
            bresenham_line(image, x_img[i], y_img[i], x_img[i + 1], y_img[i + 1], increment=10)

    return successful_paths, image  # type: ignore[return-value]


def generate_dla(
    num_particles: int = 5000,
    size: int = 256,
    seed_position: Optional[Tuple[int, int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a DLA (Diffusion-Limited Aggregation) fractal cluster.

    DLA creates branching fractal patterns by simulating particles that
    randomly walk until they stick to the growing cluster.

    Parameters
    ----------
    num_particles : int, optional
        Number of particles to aggregate, default is 5000
    size : int, optional
        Grid size, default is 256
    seed_position : tuple, optional
        Initial seed position (x, y), defaults to center

    Returns
    -------
    dla : np.ndarray
        2D array representing the DLA cluster

    Examples
    --------
    >>> from fracDimPy import generate_dla
    >>> dla = generate_dla(num_particles=3000)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(dla, cmap='binary')
    >>> plt.show()

    Notes
    -----
    DLA clusters exhibit fractal geometry with dimension typically
    between 1.66-1.71 in 2D.

    Algorithm:
    - Particles are launched from a circle around the cluster
    - They perform random walks until they stick or escape
    - Launch radius grows as cluster grows (killing radius prevents escapes)
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=bool)

    if seed_position is None:
        seed_position = (size // 2, size // 2)

    center_x, center_y = seed_position
    grid[center_y, center_x] = True

    # Track maximum radius of cluster
    max_radius = 1.0

    for _ in range(num_particles):
        # Set launch and kill radii
        launch_radius = max_radius + 5
        kill_radius = launch_radius + 20  # Outer boundary

        # Check if cluster is too large
        if launch_radius > size // 2 - 5:
            break  # Stop if approaching boundary

        # Launch particle from random angle
        angle = 2 * np.pi * rng.random()
        x = int(center_x + launch_radius * np.cos(angle))
        y = int(center_y + launch_radius * np.sin(angle))

        # Random walk
        max_steps = 10000  # Prevent infinite loops

        for step in range(max_steps):
            # Check if particle escaped
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist > kill_radius:
                break  # Particle escaped, launch new one

            # Check boundaries
            if x < 1 or x >= size - 1 or y < 1 or y >= size - 1:
                break

            # Check 4-connected neighbors
            if grid[y - 1, x] or grid[y + 1, x] or grid[y, x - 1] or grid[y, x + 1]:
                # Stick particle
                grid[y, x] = True

                # Update maximum radius
                particle_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if particle_dist > max_radius:
                    max_radius = particle_dist
                break

            # Check 8-connected neighbors (optional, for denser clusters)
            if grid[y - 1, x - 1] or grid[y - 1, x + 1] or grid[y + 1, x - 1] or grid[y + 1, x + 1]:
                grid[y, x] = True

                particle_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if particle_dist > max_radius:
                    max_radius = particle_dist
                break

            # Random walk step (8 directions)
            direction = rng.integers(0, 8)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            dx, dy = moves[direction]
            x += dx
            y += dy

    return grid.astype(int)


def generate_menger_sponge(level: int = 3, size: int = 27) -> np.ndarray:
    """
    Generate a Menger sponge fractal (3D Sierpinski carpet).

    Parameters
    ----------
    level : int, optional
        Number of iterations, default is 3
    size : int, optional
        Should be 3^level, default is 27

    Returns
    -------
    sponge : np.ndarray
        3D array representing the Menger sponge

    Examples
    --------
    >>> from fracDimPy import generate_menger_sponge
    >>> sponge = generate_menger_sponge(level=3)
    >>> print(f"Shape: {sponge.shape}")

    Notes
    -----
    The Menger sponge is the 3D analog of the Sierpinski carpet.
    It has fractal dimension log(20)/log(3) ≈ 2.727
    """
    cube = np.ones((size, size, size), dtype=bool)

    def remove_centers(arr, depth):
        if depth == 0:
            return

        s = arr.shape[0]
        step = s // 3

        # Remove cross-shaped pattern through center
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Remove if two or more coordinates are at center (1)
                    if (i == 1 and j == 1) or (j == 1 and k == 1) or (i == 1 and k == 1):
                        arr[
                            i * step : (i + 1) * step,
                            j * step : (j + 1) * step,
                            k * step : (k + 1) * step,
                        ] = False

        # Recursively process remaining sub-cubes
        if depth > 1:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if arr[i * step, j * step, k * step]:  # Only if not removed
                            sub = arr[
                                i * step : (i + 1) * step,
                                j * step : (j + 1) * step,
                                k * step : (k + 1) * step,
                            ]
                            remove_centers(sub, depth - 1)

    remove_centers(cube, level)

    return cube.astype(int)
