# Projekt: Detekcia hrán a čiar

Tento projekt implementuje vlastné algoritmy na detekciu hrán a čiar pomocou Pythonu a OpenCV. Hlavným zameraním je použitie Sobelových filtrov na detekciu hrán a vlastná implementácia Houghovej transformácie na detekciu čiar v obrázku.

## Funkcie

1. **Detekcia hrán**:
   - Používa Sobelove filtre na výpočet gradientných magnitúd.
   - Vlastná implementácia prahovania hrán s hysterezou.

2. **Detekcia čiar**:
   - Vlastná implementácia Houghovej transformácie.
   - Nastaviteľné parametre na detekciu čiar pomocou posuvníkov v OpenCV.

3. **Interaktívne GUI**:
   - Ladenie parametrov v reálnom čase pomocou posuvníkov v OpenCV.
   - Vizualizácia výsledkov detekcie hrán a čiar.

## Funkcie v projekte

### `nothing(x)`

Táto funkcia slúži ako zástupná funkcia pre posuvníky v OpenCV. Nevykonáva žiadnu akciu pri zmene hodnoty posuvníka.

### `sobel_filter(image)`

Aplikuje Sobelove filtre na obrázok na výpočet gradientov v osiach x a y.

- **Vstup**: Grayscale obrázok (numpy.ndarray).
- **Výstup**: Dva numpy polia reprezentujúce gradienty v osiach x a y.

### `canny_with_sobel(gray, t1, t2)`

Vykonáva detekciu hrán pomocou vlastnej metódy založenej na Sobelových filtroch a prahovaní s hysterezou.

- **Vstup**: Grayscale obrázok (numpy.ndarray), dolný prah (t1), horný prah (t2).
- **Výstup**: Binárny obrázok s detekovanými hranami.

### `hough_lines_p_custom(edges, rho_res, theta_res, threshold, min_line_length, max_line_gap)`

Vlastná implementácia pravdepodobnostnej Houghovej transformácie na detekciu úsečiek.

- **Vstup**: Binárny obrázok s hranami, rozlíšenie rho (rho_res), rozlíšenie theta (theta_res), prah hlasov (threshold), minimálna dĺžka úsečky (min_line_length), maximálna medzera medzi úsečkami (max_line_gap).
- **Výstup**: Zoznam detekovaných úsečiek, kde každá úsečka je reprezentovaná ako ((x1, y1), (x2, y2)).

