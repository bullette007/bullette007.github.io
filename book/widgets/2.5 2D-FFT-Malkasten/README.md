# 2.5 · 2D-FFT-Malkasten

`index.html` direkt im Browser öffnen. Das Widget ist eine eigenständige HTML-Datei mit eingebauter JavaScript-FFT, ohne externe Bibliotheken, Installation oder Internetverbindung.

## Bedienung

- **Sprache:** Englisch ist Standard. Über **Deutsch / English** in der Kopfzeile umschalten; `lang=de` öffnet die deutsche Oberfläche direkt.
- **Frequenz-Malmodus:** „Paint spectrum directly“ bearbeitet die komplexen Spektralkoeffizienten wie bisher. „Paint filter mask (0 / 1)“ malt stattdessen eine separate binäre Maske: **Pinsel = 1 (durchlassen), Radierer = 0 (sperren)**. Das Ergebnis ist `G = M · G_Basis`; vorhandene Amplituden und Phasen der Basis bleiben erhalten, auch wenn Frequenzen zunächst gesperrt und später wieder freigegeben werden.
- **Maskenbasis:** Beim Wechsel in den Maskenmodus wird das aktuell sichtbare Spektrum als Basis übernommen; die Maske startet mit Einsen. Ändert man danach ein Filter-Preset oder einen Filterregler, wird die Basis aus der Quelle neu berechnet, die gezeichnete Maske bleibt erhalten: `G = M · H · G_Quelle`. Beim Wechsel zurück in den Direktmodus wird das aktuelle Ergebnis direkt bearbeitbar. Neue Testbilder, Bild-Uploads und Bearbeitungen im Ortsraum setzen die Maske auf Einsen zurück.
- **Maskenanzeige:** „Show filter mask“ zeigt rechts Weiß = 1 und Schwarz = 0 statt des Ergebnisspektrums. „Fill mask with 1 / 0“ lässt alle Frequenzen durch bzw. sperrt alle, um anschließend gezielt Frequenzen freizumalen. Beide Ansichten sind im Maskenmodus bemalbar; konjugierte Partner bleiben synchron. Rückgängig und JSON-Speicherung umfassen Maske und Basis. „Restore spectrum“ setzt auch die Maske auf Einsen zurück.
- **Links im Bild malen:** Pinsel setzt die gewählte Helligkeit, Radierer setzt 0. Das Spektrum aktualisiert sich live. Das bearbeitete Bild wird zur neuen Filterquelle.
- **Rechts im Log-Magnitudenspektrum malen:** Pinsel hebt die Amplitude an, Radierer entfernt Frequenzen; das Bild wird live rücktransformiert. Die Phase bestehender Koeffizienten bleibt erhalten. Neue Frequenzen erhalten Phase 0.
- **Reelle Bilder:** Konjugierte Frequenzpaare werden automatisch gemeinsam bearbeitet. Die Frequenzfläche ist periodisch; der Pinsel läuft über die Ränder hinweg.
- **Filter-Presets:** Gauß-Tiefpass, Gauß-Hochpass, Kreisring-Bandpass, Richtungsfilter, Pillbox und Bewegungs-Rect. Regleränderungen wenden den Filter neu auf die Quelle an; Filter werden nicht kumuliert. Direkte freie Spektraländerungen werden dabei ersetzt; gezeichnete Filtermasken bleiben erhalten.
- **Pillbox:** Kreisförmige Maske **im Frequenzraum**, entsprechend `fourier_filtering(r)` in `02_Basics.md`. Nicht mit der Fourier-Transformierten einer räumlichen Pillbox-PSF verwechseln. `r` ist der Anteil der axialen Nyquist-Frequenz, der Maskenradius beträgt `r × 128` Frequenzbins.
- **Bewegungs-Rect:** Normierter, zentrierter Linienkernel im Ortsraum, dessen FFT mit dem Quellspektrum multipliziert wird. Länge und Bewegungsrichtung sind einstellbar.
- **fftshift:** Verschiebt nur die Spektrumdarstellung und die Mal-Koordinaten. Das Bild bleibt identisch. An: DC in der Mitte; aus: DC links oben, höchste axiale Frequenzen bei Index 128.
- **Auto-Kontrast:** Bildminimum/-maximum werden auf Schwarz/Weiß abgebildet. Ohne Automatik ist die Anzeigeskala [0, 1]. Berechnete negative Werte und Überschwinger werden intern nicht abgeschnitten. Der tatsächliche Wertebereich steht unter dem Bild. Das Log-Spektrum hat eine feste Skala `log(1 + |G|) / log(1 + 256²)`.
- **Eigene Bilder:** Lokale Bilder werden in Graustufen auf 256 × 256 eingepasst; ihr Seitenverhältnis bleibt erhalten, freie Ränder sind schwarz.
- **Rückgängig:** Bis zu zwölf Bearbeitungsschritte. Ein Pinselstrich ist ein Schritt.
- **Speichern:** PNG exportiert die angezeigte Bilddarstellung. „Zustand speichern/laden“ sichert Quelle, komplexes Spektrum und Einstellungen vollständig als JSON, einschließlich eigener Bilder und Zeichnungen.
- **Tastatur:** B = Pinsel, E = Radierer, S = fftshift, R = Reset, Strg/Cmd+Z = Rückgängig. Pfeiltasten bedienen fokussierte Regler. Rechte Maustaste radiert; Touch/Stift werden unterstützt.

## Einstellungen per URL

Die URL wird bei Änderungen aktualisiert. Sie reproduziert Testbild, Filter, Sprache, Malmodus und Regler. **Eigene Bilddaten, Zeichnungen und gemalte Masken sind nicht Teil der URL**; dafür die Zustandsdatei verwenden. `scene=custom` fällt beim Öffnen eines Links auf das eingebaute Testmotiv zurück. Gespeicherte JSON-Zustände der bisherigen Version können weiterhin geladen werden.

| Parameter | Werte / Bedeutung | Standard |
| --- | --- | --- |
| `scene` | `target`, `stripes`, `checker`, `points`, `blank` | `target` |
| `filter` | `none`, `low`, `high`, `band`, `direction`, `pillbox`, `motion` | `none` |
| `radius` | 0.02 … 0.5, relativ zur axialen Nyquist-Frequenz | `0.12` |
| `width` | 0.02 … 0.4, volle Breite des Kreisrings | `0.08` |
| `length` | 1 … 48 Pixel, Bewegungslänge | `15` |
| `angle` | 0 … 175 Grad, im Bild nach unten positiv | `0` |
| `brushSize` | 1 … 24 Pixel, Pinselradius | `5` |
| `brightness` | 0 … 1, Helligkeit im Bild | `1` |
| `amplitude` | 0.001 … 0.03, spektraler Betrag geteilt durch 256² | `0.008` |
| `shift` | `1` = fftshift an, `0` = aus | `1` |
| `contrast` | `1` = automatischer Bildkontrast | `1` |
| `theme` | `light`, `dark` | `light` |
| `lang` | `en`, `de` | `en` |
| `paintMode` | `direct` = Spektrum direkt, `mask` = Filtermaske malen | `direct` |
| `maskView` | `1` = im Maskenmodus die Maske anzeigen, `0` = Ergebnisspektrum | `0` |

Beispiel für den Pillbox-Tiefpass aus dem Skript:

```text
index.html?scene=target&filter=pillbox&radius=0.08&shift=1&theme=light
```

## Einbindung im Notebook / Web-Buch

Bei einem lokalen Webserver im Projektordner auf Port 8000:

```python
from IPython.display import IFrame, display

display(IFrame(
    src="http://127.0.0.1:8000/widgets/"
        "2.5%202D-FFT-Malkasten/index.html"
        "?filter=pillbox&radius=0.08&shift=1",
    width="100%",
    height=1100,
))
```

Im veröffentlichten HTML-Buch:

```html
<iframe
  src="widgets/2.5%202D-FFT-Malkasten/index.html?filter=pillbox&amp;radius=0.08"
  title="2D-FFT-Malkasten: Bild und Spektrum bearbeiten"
  width="100%"
  height="1100"
  style="border:0"
  loading="lazy">
</iframe>
```

Der Pfad muss relativ zur Buchseite passen; die HTML-Datei muss mit veröffentlicht werden. Das responsive Layout wechselt bei wenig Platz zu gestapelten Ansichten. Für die Vorlesung Vollbild verwenden. Für PDF-Screenshots stehen die Selektoren `#image`, `#spectrum` und `.views` zur Verfügung.

## Drei kurze Vorlesungsversuche

1. **Sinus selbst bauen:** Bild leeren, Pinselradius 1, bei aktivem fftshift etwas neben DC ins Spektrum klicken. Welche Richtung haben die Streifen? Was ändert sich, wenn der Frequenzpunkt weiter nach außen wandert?
2. **Harte Grenze, Ringing:** Testmotiv laden, Pillbox auswählen, `radius=0.08`. Mit dem Gauß-Tiefpass vergleichen: Welche Filtergrenze erzeugt Überschwinger?
3. **fftshift verstehen:** Sinusstreifen laden, fftshift aus/an schalten. Die Peaks wandern, das Bild bleibt gleich. Wo liegen Gleichanteil und Nyquist-Frequenzen jeweils?
