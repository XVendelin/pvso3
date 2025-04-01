import numpy as np
import cv2 as cv


def nothing(x):
    pass


import numpy as np


def sobel_filter(image):
    # Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    rows, cols = image.shape
    sobelx = np.zeros_like(image, dtype=np.float64)
    sobely = np.zeros_like(image, dtype=np.float64)

    # Apply Sobel filters
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            sobelx[i, j] = np.sum(region * Kx)
            sobely[i, j] = np.sum(region * Ky)

    return sobelx, sobely


def canny_with_sobel(gray, t1, t2):
    sobelx, sobely = sobel_filter(gray)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_magnitude = (255 * gradient_magnitude / np.max(gradient_magnitude)).astype(np.uint8)

    # Apply thresholding (Hysteresis)
    edges = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    strong_edges = gradient_magnitude > t2
    weak_edges = (gradient_magnitude >= t1) & (gradient_magnitude < t2)

    edges[strong_edges] = 255
    edges[weak_edges] = 128  # Potential edges

    return edges


def hough_lines_p_custom(edges, rho_res=1, theta_res=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10):
    # Step 1: Get edge points
    y_idxs, x_idxs = np.where(edges > 0)
    num_points = len(x_idxs)

    # Step 2: Define θ and ρ ranges
    theta_vals = np.arange(0, np.pi, theta_res)  # 0 to π (avoids duplicate angles)
    max_rho = int(np.hypot(edges.shape[0], edges.shape[1]))  # Max possible rho
    rho_vals = np.arange(-max_rho, max_rho, rho_res)

    # Step 3: Accumulator (ρ, θ) voting
    accumulator = np.zeros((len(rho_vals), len(theta_vals)), dtype=np.int32)
    theta_cos = np.cos(theta_vals)
    theta_sin = np.sin(theta_vals)

    # Step 4: Cast votes
    for i in range(num_points):
        x, y = x_idxs[i], y_idxs[i]
        for t_idx, (cos_t, sin_t) in enumerate(zip(theta_cos, theta_sin)):
            rho = int(x * cos_t + y * sin_t)
            rho_idx = np.argmin(np.abs(rho_vals - rho))
            accumulator[rho_idx, t_idx] += 1

    # Step 5: Find peaks in accumulator
    detected_lines = []
    rho_theta_idxs = np.argwhere(accumulator > threshold)

    for rho_idx, theta_idx in rho_theta_idxs:
        rho = rho_vals[rho_idx]
        theta = theta_vals[theta_idx]

        # Convert (ρ, θ) to a standard line equation
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))

        detected_lines.append(((x1, y1), (x2, y2), theta))

    # Step 6: Extract edge points along detected lines
    line_segments = []
    for (x1, y1), (x2, y2), theta in detected_lines:
        segment = []
        for i in range(num_points):
            x, y = x_idxs[i], y_idxs[i]
            if abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / max(1, np.hypot(x2 - x1, y2 - y1)) < 2:
                segment.append((x, y))

        if len(segment) < 2:
            continue

        # Determine if the line is more horizontal or vertical
        dx = abs(segment[-1][0] - segment[0][0])
        dy = abs(segment[-1][1] - segment[0][1])

        if dx >= dy:
            segment.sort(key=lambda p: p[0])  # Sort by x for horizontal lines
        else:
            segment.sort(key=lambda p: p[1])  # Sort by y for vertical lines

        # Merge close segments
        merged_segments = []
        start = segment[0]
        prev = start

        for point in segment[1:]:
            if np.hypot(point[0] - prev[0], point[1] - prev[1]) > max_line_gap:
                if np.hypot(prev[0] - start[0], prev[1] - start[1]) >= min_line_length:
                    merged_segments.append((start, prev))
                start = point
            prev = point

        if np.hypot(prev[0] - start[0], prev[1] - start[1]) >= min_line_length:
            merged_segments.append((start, prev))

        line_segments.extend(merged_segments)

    return line_segments


# Načítanie obrázka
image_path = ("img_und6.jpg")  # Upravte cestu k obrázku
img = cv.imread(image_path)
if img is None:
    print("Error: Could not load image.")
    exit()

# Konverzia na odtiene sivej
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.dot(img[..., :3], [0.114, 0.587, 0.2989]).astype(np.uint8)

# Vytvorenie okna s trackbarmi
cv.namedWindow('img')
cv.createTrackbar('threshold1', 'img', 150, 255, nothing)
cv.createTrackbar('threshold2', 'img', 150, 255, nothing)
cv.createTrackbar('min_line_length', 'img', 50, 500, nothing)
cv.createTrackbar('max_line_gap', 'img', 10, 100, nothing)
cv.createTrackbar('theta', 'img', 2, 90, nothing)
cv.createTrackbar('rho', 'img', 2, 50, nothing)
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'img', 0, 1, nothing)

while True:
    # Načítanie hodnôt z trackbarov
    t1 = cv.getTrackbarPos('threshold1', 'img')
    t2 = cv.getTrackbarPos('threshold2', 'img')
    min_line_length = cv.getTrackbarPos('min_line_length', 'img')
    max_line_gap = cv.getTrackbarPos('max_line_gap', 'img')
    theta = cv.getTrackbarPos('theta', 'img')
    rho = cv.getTrackbarPos('rho', 'img')
    s = cv.getTrackbarPos(switch, 'img')

    # Kopírovanie originálneho obrázka
    output = img.copy()

    # Detekcia hrán pomocou Cannyho detektora
    # použiť vlastný filter prednáška 5

    edges = canny_with_sobel(gray, t1, t2)
    cv.imshow('gray', edges)

    if s == 1:
        # Houghova transformácia na detekciu čiar
        # toto treba prepísať
        #lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)

        lines = hough_lines_p_custom(edges, rho, np.pi/theta, 50, min_line_length, max_line_gap)

        if lines is not None:
            for (x1, y1), (x2, y2) in lines:
                cv.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Zobrazenie výsledku
    cv.imshow('img', output)

    # Ukončenie programu po stlačení medzerníka
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

cv.destroyAllWindows()