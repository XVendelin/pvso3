import numpy as np
import cv2 as cv


def nothing(x):
    pass

def canny_with_sobel(gray, t1, t2):
    # Compute Sobel gradients
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # X gradient
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # Y gradient

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))  # Normalize

    # Apply thresholding (Hysteresis)
    edges = np.zeros_like(gradient_magnitude)
    strong_edges = gradient_magnitude > t2
    weak_edges = (gradient_magnitude >= t1) & (gradient_magnitude < t2)

    edges[strong_edges] = 255
    edges[weak_edges] = 128  # Potential edges

    return edges




# Načítanie obrázka
image_path = "img_und1.jpg"  # Upravte cestu k obrázku
img = cv.imread(image_path)
if img is None:
    print("Error: Could not load image.")
    exit()

# Konverzia na odtiene sivej
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Použitie Gaussovho rozostrenia na zníženie šumu
gray = cv.GaussianBlur(gray, (5, 5), 0)

# Vytvorenie okna s trackbarmi
cv.namedWindow('img')
cv.createTrackbar('threshold1', 'img', 50, 255, nothing)
cv.createTrackbar('threshold2', 'img', 150, 255, nothing)
cv.createTrackbar('min_line_length', 'img', 50, 500, nothing)
cv.createTrackbar('max_line_gap', 'img', 10, 100, nothing)
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'img', 1, 1, nothing)

while True:
    # Načítanie hodnôt z trackbarov
    t1 = cv.getTrackbarPos('threshold1', 'img')
    t2 = cv.getTrackbarPos('threshold2', 'img')
    min_line_length = cv.getTrackbarPos('min_line_length', 'img')
    max_line_gap = cv.getTrackbarPos('max_line_gap', 'img')
    s = cv.getTrackbarPos(switch, 'img')

    # Kopírovanie originálneho obrázka
    output = img.copy()

    if s == 1:
        # Detekcia hrán pomocou Cannyho detektora
        # použiť vlastný filter prednáška 5
        edges = canny_with_sobel(gray, t1, t2)

        # Houghova transformácia na detekciu čiar
        # toto treba prepísať
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Zobrazenie výsledku
    cv.imshow('img', output)

    # Ukončenie programu po stlačení medzerníka
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

cv.destroyAllWindows()