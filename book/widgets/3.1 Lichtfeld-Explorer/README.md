# 3.1 · Lichtfeld-Explorer

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget ohne externe Bibliotheken, mit Deutsch/Englisch, Hell/Dunkel, Zustandslinks und HTML-Export. Standard: Englisch und dunkles Farbschema; Deutsch mit `?lang=de`.

## Lernweg

1. **Mikrolinsen-Rohbild:** Ein 672 × 504-Rohbild enthält 96 × 72 Mikrobilder mit jeweils 7 × 7 Winkelpixeln. Dieselbe relative Pixelposition wird unter allen Mikrolinsen markiert.
2. **Subaperturbild:** Diese Messwerte bilden genau ein 96 × 72-Bild. Mit `j` und `k` oder im vergrößerten Mikrobild die Ansicht wechseln und die Parallaxe beobachten.
3. **Shift and Add:** Mehrere Ansichten werden bilinear verschoben und gemittelt. Die drei Fokus-Presets richten die ferne, mittlere bzw. nahe Karte aus. Eine zusätzliche 50/50-Überlagerung macht die Verschiebung der gewählten Ansicht gegenüber der Zentralansicht sichtbar.
4. **Virtuelle Blende und Rauschen:** Radius 0 nutzt nur die Zentralansicht, Radius 3 alle 49 Ansichten. Optionales deterministisches Messrauschen veranschaulicht den Unterschied zwischen Einzelansicht und Mittelung.

Ein Klick ins Rohbild wählt Mikrolinse und Winkelpixel; ein Klick ins SAI wählt die räumliche Probe. Dieselbe Probe ist in beiden Bildern orange markiert. Alle Koordinaten sind 1-basiert. Die Winkelmatrix ist per Tab und Pfeiltasten bedienbar; alle Koordinaten sind außerdem über native Schieberegler zugänglich.

## Modell

Der komplette 4D-Datensatz wird lokal aus drei texturierten, sich verdeckenden Ebenen erzeugt: Disparität −1,5 (Fern/blau), 0 (Mitte/grün), +1,5 (Nah/orange) SAI-Pixel pro Winkelschritt. Es sind synthetische Daten. Rohbild und Rekonstruktion verwenden identische Samples; die Rekonstruktion verwendet keine Szenengeometrie.

Zentrierte Winkelkoordinaten: `u = j − 4`, `v = k − 4`.

```text
L(m,n,j,k) = g(7(m−1)+j, 7(n−1)+k)
I_s(x,y) = mean_(u,v in A_r) L(x+s*u, y+s*v, u+4, v+4)
A_r = {−r,…,r}²
```

Die sichtbare Bildverschiebung beträgt `(-s*u, -s*v)`. `s` ist eine diskrete Scherung in Pixeln pro Winkelschritt, **nicht** das dimensionslose `α = b′/b` aus dem Skript. Der gemeinsame Streckfaktor wird wie im Shift-and-Add-Abschnitt vernachlässigt. Außerhalb des Bilds liegende Samples werden ausgeschlossen, die Mittelung wird pro Pixel normiert. Die Zentralansicht bleibt immer gültig. An Verdeckungskanten können auch bei passender Scherung Mischungen auftreten.

Additives gaußförmiges Rauschen mit festem Seed 17, unabhängig je Ansicht, Ort und Farbkanal; intern ungeclippt, Anzeige auf 0–255 begrenzt. Keine Photonen-, Beugungs- oder Vignettierungssimulation. Die quadratisch gepackten Mikrobilder repräsentieren eine ideale kalibrierte plenoptische Kamera.

## Einbettung

```html
<iframe src="widgets/3.1%20Lichtfeld-Explorer/index.html?lang=de"
        title="Lichtfeld-Explorer" width="100%" height="2200"
        style="border:0" loading="lazy"></iframe>
```

Auf schmalen Bildschirmen stehen die Panels untereinander; der iframe ist deshalb scrollbar. Das Kapitel bindet den Explorer nach der Herleitung von Shift and Add ein.

URL-Parameter: `j`, `k` (1–7), `m` (1–96), `n` (1–72), `focus` (−2…2, Schritt 0,05), `radius` (0–3), `noise` (0–25), `marks`, `dark` (`true`/`false`), `lang` (`de`/`en`). URL-Parameter überschreiben den eingebetteten Exportzustand. Reset erhält Sprache und Farbschema.

Beispiele:

```text
index.html?lang=de&j=7&k=4&focus=1.5
index.html?lang=de&radius=0&noise=20
```

## Prüfung

Mit Playwright und Chromium:

```text
python "widgets/3.1 Lichtfeld-Explorer/verify_browser.py"
```

Prüft Rohbild/SAI-Zuordnung, Auswahl, numerische Shift-and-Add-Ergebnisse gegen unabhängige Referenzrechnung, Fokuswirkung, Blende, Rauschen, URL-Zustände, Sprache, Theme, Offline-Export und mobile Überlaufgrenzen.
