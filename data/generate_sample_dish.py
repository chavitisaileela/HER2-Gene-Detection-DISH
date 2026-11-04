"""
generate_sample_dish.py
Creates realistic synthetic brightfield Dual In Situ Hybridization (DISH) images
replicating the characteristics described in Micaraseth et al. (Intelligence-Based Medicine 2026):
- Pale blue hematoxylin-stained cell nuclei (50-100 px diameter)
- Black chromogenic dots (HER2 gene signals)
- Red chromogenic dots (CEP17 centromere signals)
- Slight background texture and chromatic variation
"""

import os
import numpy as np
import cv2

def generate_dish_image(output_path, num_cells=35, mode="positive", img_size=(1500, 1500), seed=42):
    np.random.seed(seed)
    h, w = img_size
    
    # Background: off-white brightfield microscopy illumination with slight vignetting
    bg = np.ones((h, w, 3), dtype=np.uint8) * 242
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 215, 255).astype(np.uint8)
    
    # Add subtle stromal background streaks
    for _ in range(12):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (pt1[0] + np.random.randint(-200, 200), pt1[1] + np.random.randint(-200, 200))
        cv2.line(bg, pt1, pt2, (225, 220, 230), thickness=np.random.randint(2, 6))

    img = bg.copy()
    
    cells = []
    # Place non-overlapping or minimally touching nuclei
    attempts = 0
    while len(cells) < num_cells and attempts < 1000:
        attempts += 1
        cx = np.random.randint(100, w - 100)
        cy = np.random.randint(100, h - 100)
        rx = np.random.randint(32, 50)
        ry = np.random.randint(30, 48)
        angle = np.random.randint(0, 180)
        
        # Check collision with existing nuclei
        collision = False
        for ocx, ocy, oradius in cells:
            dist = np.hypot(cx - ocx, cy - ocy)
            if dist < (rx + oradius + 15):
                collision = True
                break
        if collision:
            continue
            
        cells.append((cx, cy, max(rx, ry)))
        
        # Draw nucleus: Pale blue/hematoxylin hue (BGR: higher Blue, moderate Green/Red)
        # Nucleus interior
        nucleus_overlay = img.copy()
        cv2.ellipse(nucleus_overlay, (cx, cy), (rx, ry), angle, 0, 360, (200, 175, 140), -1)
        cv2.ellipse(nucleus_overlay, (cx, cy), (rx, ry), angle, 0, 360, (170, 140, 110), 2) # border
        cv2.addWeighted(nucleus_overlay, 0.75, img, 0.25, 0, img)
        
        # Determine signals based on mode
        if mode == "positive":
            # Amplified: 4 to 8 black HER2 dots, 1 to 2 red CEP17 dots
            num_her2 = np.random.randint(4, 9)
            num_cep17 = np.random.randint(1, 3)
        elif mode == "negative":
            # Non-amplified: 1 to 2 black HER2 dots, 1 to 2 red CEP17 dots
            num_her2 = np.random.randint(1, 3)
            num_cep17 = np.random.randint(1, 3)
        else: # borderline
            num_her2 = np.random.choice([3, 4])
            num_cep17 = 2
            
        # Draw red CEP17 signals (Red in BGR: [B < 60, G < 60, R > 190])
        for _ in range(num_cep17):
            rad = np.random.uniform(0, 0.7)
            theta = np.random.uniform(0, 2 * np.pi)
            sx = int(cx + rad * rx * np.cos(theta))
            sy = int(cy + rad * ry * np.sin(theta))
            dot_r = np.random.randint(3, 5)
            # Red color with soft edge
            cv2.circle(img, (sx, sy), dot_r, (40, 30, 215), -1)
            cv2.circle(img, (sx, sy), dot_r + 1, (60, 50, 180), 1)

        # Draw black HER2 signals (Black in BGR: [B < 75, G < 75, R < 75])
        for _ in range(num_her2):
            rad = np.random.uniform(0, 0.75)
            theta = np.random.uniform(0, 2 * np.pi)
            sx = int(cx + rad * rx * np.cos(theta))
            sy = int(cy + rad * ry * np.sin(theta))
            dot_r = np.random.randint(3, 5)
            # Black/very dark brown dot
            cv2.circle(img, (sx, sy), dot_r, (25, 25, 25), -1)
            cv2.circle(img, (sx, sy), dot_r + 1, (45, 45, 45), 1)

    # Slight Gaussian smoothing for biological imaging realism
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Generated {mode} DISH sample with {len(cells)} cells -> {output_path}")

if __name__ == "__main__":
    out_dir = "C:/Users/chavi/.gemini/antigravity/scratch/HER2_DISH_Quantification/data"
    generate_dish_image(os.path.join(out_dir, "sample_positive_patient_c001.png"), num_cells=30, mode="positive", seed=101)
    generate_dish_image(os.path.join(out_dir, "sample_negative_patient_c002.png"), num_cells=28, mode="negative", seed=202)
    generate_dish_image(os.path.join(out_dir, "sample_borderline_patient_c003.png"), num_cells=25, mode="borderline", seed=303)
