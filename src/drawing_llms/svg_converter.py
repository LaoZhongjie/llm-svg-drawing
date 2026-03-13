"""Bitmap-to-SVG conversion logic."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def compress_hex_color(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f"#{r // 17:x}{g // 17:x}{b // 17:x}"
    return hex_color


def extract_features_by_scale(img_np: np.ndarray, num_colors: int = 16) -> list[dict]:
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)

    hierarchical_features = []
    _, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]

    center_x, center_y = width / 2, height / 2

    for color in sorted_colors:
        color_mask = cv2.inRange(quantized, color, color)
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        hex_color = compress_hex_color(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")

        color_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:
                continue

            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue

            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            dist_from_center = np.sqrt(
                ((cx - center_x) / width) ** 2 + ((cy - center_y) / height) ** 2
            )

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])

            importance = area * (1 - dist_from_center) * (1 / (len(approx) + 1))
            color_features.append(
                {
                    "points": points,
                    "color": hex_color,
                    "area": area,
                    "importance": importance,
                    "point_count": len(approx),
                    "original_contour": approx,
                }
            )

        color_features.sort(key=lambda x: x["importance"], reverse=True)
        hierarchical_features.extend(color_features)

    hierarchical_features.sort(key=lambda x: x["importance"], reverse=True)
    return hierarchical_features


def simplify_polygon(points_str: str, simplification_level: int) -> str:
    if simplification_level == 0:
        return points_str

    points = points_str.split()

    if simplification_level == 1:
        return " ".join(
            [
                f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}"
                for p in points
            ]
        )

    if simplification_level == 2:
        return " ".join(
            [
                f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                for p in points
            ]
        )

    if simplification_level == 3:
        if len(points) <= 4:
            return " ".join(
                [
                    f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                    for p in points
                ]
            )
        step = min(2, len(points) // 3)
        reduced_points = [points[i] for i in range(0, len(points), step)]
        if len(reduced_points) < 3:
            reduced_points = points[:3]
        if points[-1] not in reduced_points:
            reduced_points.append(points[-1])
        return " ".join(
            [
                f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                for p in reduced_points
            ]
        )

    return points_str


def bitmap_to_svg_layered(
    image: Image.Image,
    max_size_bytes: int = 10000,
    resize: bool = True,
    target_size: tuple[int, int] = (384, 384),
    adaptive_fill: bool = True,
    num_colors: int | None = None,
) -> str:
    if num_colors is None:
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]

        if pixel_count < 65536:
            num_colors = 8
        elif pixel_count < 262144:
            num_colors = 12
        else:
            num_colors = 16

    if resize:
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
    else:
        original_size = image.size

    img_np = np.array(image)
    height, width = img_np.shape[:2]

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
        bg_hex_color = compress_hex_color(
            f"#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}"
        )
    else:
        bg_hex_color = "#fff"

    orig_width, orig_height = original_size
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" '
        f'height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    )
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = "</svg>"

    base_size = len((svg_base + svg_footer).encode("utf-8"))
    available_bytes = max_size_bytes - base_size
    if available_bytes <= 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'
        )

    features = extract_features_by_scale(img_np, num_colors=num_colors)

    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            feature_svg = (
                f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            )
            if len((svg + feature_svg + svg_footer).encode("utf-8")) > max_size_bytes:
                break
            svg += feature_svg
        svg += svg_footer
        return svg

    feature_sizes = []
    for feature in features:
        feature_sizes.append(
            {
                "original": len(
                    f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level1": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level2": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level3": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
            }
        )

    svg = svg_base
    bytes_used = base_size
    added_features = set()

    for i, feature in enumerate(features):
        feature_svg = (
            f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
        )
        feature_size = feature_sizes[i]["original"]
        if bytes_used + feature_size <= max_size_bytes:
            svg += feature_svg
            bytes_used += feature_size
            added_features.add(i)

    for level in range(1, 4):
        for i, feature in enumerate(features):
            if i in added_features:
                continue
            feature_size = feature_sizes[i][f"level{level}"]
            if bytes_used + feature_size <= max_size_bytes:
                feature_svg = (
                    f'<polygon points="{simplify_polygon(feature["points"], level)}" '
                    f'fill="{feature["color"]}" />\n'
                )
                svg += feature_svg
                bytes_used += feature_size
                added_features.add(i)

    svg += svg_footer
    final_size = len(svg.encode("utf-8"))
    if final_size > max_size_bytes:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'
        )
    return svg
