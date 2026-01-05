import cv2
import numpy as np
import os


# Edge detection using Sobel operator
def detect_edges_sobel(img_gray, thresh=30):
    # Compute gradients in both directions
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get magnitude and direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    
    # Normalize to 8-bit range
    normalized = np.uint8(np.clip(grad_magnitude / grad_magnitude.max() * 255, 0, 255))
    
    # Threshold the edges
    edge_map = np.zeros_like(normalized)
    edge_map[normalized > thresh] = 255
    
    return edge_map, normalized, grad_direction


# Scan outward from center to find lane edges
def scan_for_lane_edges(edge_img, directions, start_y, center_pos):
    h, w = edge_img.shape
    left_edge_pts = []
    right_edge_pts = []
    
    # Scan each row from start_y to bottom
    for row in range(start_y, h):
        # Scan right side
        for col in range(center_pos, w):
            if edge_img[row, col] > 0:
                angle_val = directions[row, col]
                right_edge_pts.append((col, row))
                break
        
        # Scan left side
        for col in range(center_pos, -1, -1):
            if edge_img[row, col] > 0:
                left_edge_pts.append((col, row))
                break
    
    return left_edge_pts, right_edge_pts


# Fit line to points using Hough transform and slope constraints
def fit_line_to_points(pts, img_dims, slope_limits):
    if len(pts) < 10:
        return None
    
    h, w = img_dims
    min_s, max_s = slope_limits
    
    # Build binary image from points
    bin_img = np.zeros((h, w), dtype=np.uint8)
    for px, py in pts:
        if 0 <= px < w and 0 <= py < h:
            bin_img[py, px] = 255
    
    # Run Hough transform
    detected_lines = cv2.HoughLines(bin_img, rho=1, theta=np.pi/180, threshold=15)
    
    if detected_lines is None:
        return None
    
    # Find best line within slope range
    best_m, best_b = None, None
    max_score = 0
    
    for idx, ln in enumerate(detected_lines):
        rho_val, theta_val = ln[0]
        
        # Convert to y = mx + b form
        if abs(np.sin(theta_val)) < 1e-6:
            continue
        
        slope = -np.cos(theta_val) / np.sin(theta_val)
        intercept = rho_val / np.sin(theta_val)
        
        # Check slope constraint
        if min_s <= slope <= max_s:
            score = len(detected_lines) - idx
            if score > max_score:
                max_score = score
                best_m, best_b = slope, intercept
    
    return (best_m, best_b) if best_m is not None else None


# Calculate where two lines intersect (vanishing point)
def compute_intersection(line1, line2):
    if line1 is None or line2 is None:
        return None
    
    m1, b1 = line1
    m2, b2 = line2
    
    # Parallel lines don't intersect
    if abs(m1 - m2) < 1e-6:
        return None
    
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1
    
    return (int(x_int), int(y_int))


# Render detected lanes on image
def render_lane_overlay(img, left_ln, right_ln, vp, roi_top):
    result = img.copy()
    h, w = result.shape[:2]
    
    # Draw left lane marker
    if left_ln is not None:
        m, b = left_ln
        if vp is not None:
            vp_x, vp_y = vp
            bottom_x = int((h - 1 - b) / m) if m != 0 else 0
            cv2.line(result, (vp_x, vp_y), (bottom_x, h - 1), (0, 0, 255), 3)
        else:
            top_x = int((roi_top - b) / m) if m != 0 else 0
            bottom_x = int((h - 1 - b) / m) if m != 0 else 0
            cv2.line(result, (top_x, roi_top), (bottom_x, h - 1), (0, 0, 255), 3)
    
    # Draw right lane marker
    if right_ln is not None:
        m, b = right_ln
        if vp is not None:
            vp_x, vp_y = vp
            bottom_x = int((h - 1 - b) / m) if m != 0 else w - 1
            cv2.line(result, (vp_x, vp_y), (bottom_x, h - 1), (0, 0, 255), 3)
        else:
            top_x = int((roi_top - b) / m) if m != 0 else w - 1
            bottom_x = int((h - 1 - b) / m) if m != 0 else w - 1
            cv2.line(result, (top_x, roi_top), (bottom_x, h - 1), (0, 0, 255), 3)
    
    # Mark vanishing point if found
    if vp is not None:
        cv2.circle(result, vp, 12, (0, 255, 255), 3)
    
    return result


# Main processing pipeline for each frame
def analyze_frame(frame):
    h, w = frame.shape[:2]
    
    # Focus on lower half of image where lanes are
    roi_start = int(h * 0.5)
    mid_x = w // 2
    
    # Prep image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Detect edges
    edges, mag, angles = detect_edges_sobel(gray_img, thresh=30)
    
    # Find candidate lane points
    left_pts, right_pts = scan_for_lane_edges(edges, angles, roi_start, mid_x)
    
    # Fit lines - left should slope down-left, right should slope down-right
    left_lane = fit_line_to_points(left_pts, (h, w), slope_limits=(-2.0, -0.3))
    right_lane = fit_line_to_points(right_pts, (h, w), slope_limits=(0.3, 2.0))
    
    # Find where lanes converge
    vanish_pt = compute_intersection(left_lane, right_lane)
    
    # Sanity check vanishing point location
    if vanish_pt is not None:
        vx, vy = vanish_pt
        if vy > h or vy < -h or vx < -w or vx > 2 * w:
            vanish_pt = None
    
    # Overlay results
    output_frame = render_lane_overlay(frame, left_lane, right_lane, vanish_pt, roi_start)
    
    return output_frame, edges


# Process all frames from a directory
def batch_process_frames(src_dir, dst_dir, prefix):
    os.makedirs(dst_dir, exist_ok=True)
    
    # Collect frame paths
    frames_to_process = []
    for frame_num in range(100):
        fname = f"{prefix}_{frame_num}.bmp"
        fpath = os.path.join(src_dir, fname)
        if os.path.exists(fpath):
            frames_to_process.append((frame_num, fpath))
    
    if not frames_to_process:
        print(f"No frames found in {src_dir} with prefix '{prefix}'")
        return
    
    print(f"Found {len(frames_to_process)} frames to process...")
    
    processed = []
    for num, path in frames_to_process:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: couldn't load {path}")
            continue
        
        # Process the frame
        result, edge_img = analyze_frame(img)
        processed.append(result)
        
        # Save outputs
        out_path = os.path.join(dst_dir, f"output_{num}.bmp")
        cv2.imwrite(out_path, result)
        
        edge_path = os.path.join(dst_dir, f"edges_{num}.bmp")
        cv2.imwrite(edge_path, edge_img)
        
        if (num + 1) % 10 == 0:
            print(f"  Progress: {num + 1} frames done")
    
    # Generate video from processed frames
    if processed:
        frame_h, frame_w = processed[0].shape[:2]
        vid_path = os.path.join(dst_dir, "output_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_writer = cv2.VideoWriter(vid_path, fourcc, 10, (frame_w, frame_h))
        
        for frm in processed:
            vid_writer.write(frm)
        
        vid_writer.release()
        print(f"Output video: {vid_path}")
    
    print(f"All outputs saved to {dst_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process first test video
    print("=" * 50)
    print("Processing TestVideo_1...")
    print("=" * 50)
    src1 = os.path.join(script_dir, "TestVideo_1")
    dst1 = os.path.join(script_dir, "Output_TestVideo_1")
    batch_process_frames(src1, dst1, "Right")
    
    # Process second test video
    print("\n" + "=" * 50)
    print("Processing TestVideo_2...")
    print("=" * 50)
    src2 = os.path.join(script_dir, "TestVideo_2")
    dst2 = os.path.join(script_dir, "Output_TestVideo_2")
    batch_process_frames(src2, dst2, "Left")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()