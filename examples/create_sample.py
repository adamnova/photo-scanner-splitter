"""
Example script to create a sample scanned image with multiple photos
for demonstration purposes
"""

import cv2
import numpy as np
import os


def create_sample_scan(output_path="examples/sample_scan.jpg"):
    """
    Create a sample scanned image with 3 photos at different angles
    """
    # Create a large "scanner bed" background (white/light gray)
    scan_width, scan_height = 2400, 1800
    background_color = (240, 240, 240)
    scan = np.full((scan_height, scan_width, 3), background_color, dtype=np.uint8)
    
    # Create 3 sample "photos" with different colors and sizes
    photos = [
        # Photo 1: Blue gradient (landscape)
        create_gradient_photo(600, 400, (100, 150, 255), (50, 80, 200)),
        # Photo 2: Green gradient (portrait)
        create_gradient_photo(400, 550, (100, 255, 150), (50, 200, 80)),
        # Photo 3: Red gradient (square)
        create_gradient_photo(500, 500, (255, 100, 100), (200, 50, 50)),
    ]
    
    # Positions and rotation angles for each photo
    positions = [
        (200, 200, 5),    # x, y, rotation_angle
        (1100, 150, -3),
        (1400, 1000, 7),
    ]
    
    # Place each photo on the scan with rotation
    for photo, (x, y, angle) in zip(photos, positions):
        # Rotate photo
        rotated = rotate_image(photo, angle)
        
        # Place on scan
        h, w = rotated.shape[:2]
        
        # Ensure it fits
        if y + h > scan_height:
            h = scan_height - y
        if x + w > scan_width:
            w = scan_width - x
        
        # Create a mask for blending
        mask = np.all(rotated[:h, :w] != background_color, axis=2)
        scan[y:y+h, x:x+w][mask] = rotated[:h, :w][mask]
    
    # Add some scanner artifacts (slight noise, shadow)
    noise = np.random.randint(-5, 5, scan.shape, dtype=np.int16)
    scan = np.clip(scan.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save the scan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, scan)
    print(f"Sample scan created: {output_path}")
    
    return output_path


def create_gradient_photo(width, height, color1, color2):
    """Create a photo with a gradient and border"""
    photo = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient
    for i in range(height):
        ratio = i / height
        color = tuple(int(c1 * (1 - ratio) + c2 * ratio) 
                     for c1, c2 in zip(color1, color2))
        photo[i, :] = color
    
    # Add a white border
    border_size = 10
    cv2.rectangle(photo, (0, 0), (width-1, height-1), (255, 255, 255), border_size)
    
    # Add some texture (simulating a photo)
    # Add random dots to simulate photo grain
    for _ in range(200):
        px, py = np.random.randint(border_size, width-border_size), np.random.randint(border_size, height-border_size)
        intensity = np.random.randint(-20, 20)
        photo[py, px] = np.clip(photo[py, px].astype(np.int16) + intensity, 0, 255).astype(np.uint8)
    
    return photo


def rotate_image(image, angle):
    """Rotate image by given angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(240, 240, 240))
    
    return rotated


if __name__ == "__main__":
    create_sample_scan()
    print("\nNow you can run:")
    print("  photo-splitter examples/sample_scan.jpg -o examples/output --no-interactive")
    print("\nOr with interactive mode:")
    print("  photo-splitter examples/sample_scan.jpg -o examples/output")
