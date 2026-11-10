# 3.2 · Kamera-Lichtfeld und Ray-Diagramm

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget ohne externe Bibliotheken, im Format der bisherigen Widgets: Deutsch/Englisch, Hell/Dunkel, responsive Darstellung, Zustandslinks und Download der vollständigen HTML-Datei mit aktuellem Zustand. Standard: Englisch und dunkles Farbschema; `?lang=de` startet auf Deutsch.

Umsetzung der Fahrplan-Idee **3.2 · Das (x,u)-Lichtfeld-Diagramm, gekoppelt an den Strahlengang**. Hier wird der **(y,v)-Schnitt** verwendet, passend zu den Abbildungen `lfInsideCamera_*` und `lfRayDiagramm_*` im Abschnitt „Light fields in a conventional camera“ von Kapitel 3.

## Die drei Ansichten

1. **Einzelner Strahl:** Orange im Strahlengang und als Punkt `(y,v)` im Lichtfeld. Die blaue Gerade zeigt die Menge aller Strahlen desselben Szenenpunkts. Der Regler `v` oder ein Klick auf Linse/Lichtfeldgerade wählt den Strahl.
2. **Sensorintegration:** Die Strahlen des Szenenpunkts treffen den realen Sensor bei unterschiedlichen Positionen, außer bei fokussierter Abbildung. Violette vertikale Linien zeigen die Integrationspfade für feste Pixel. Das ausgewählte Pixel wird links markiert; drei violette Strahlen zeigen seine rückverfolgten bildseitigen Integrationspfade.
3. **Virtueller Sensor:** `α = b′/b` verschiebt den virtuellen Sensor und schert die Integrationsgeraden im ursprünglichen Lichtfeld. Die Bündelbreite bei `b` und `b′` wird angezeigt. **Auf Szenenpunkt refokussieren** setzt `b′ = b★` und wählt das passende virtuelle Pixel: blaue Lichtfeldgerade und violette Integrationsgerade fallen zusammen.

Die Voreinstellungen **Zu nah / Im Fokus / Zu weit** setzen `g` auf 125 / 300 / 700 mm, `α` auf 1 und wählen das Pixel des Hauptstrahls. Der Sensorabstand ist `b = 90 mm`, die Standardblende `D = 32 mm`: **13,44 mm** Bündelbreite im Nah-Preset und **5,49 mm** im Fern-Preset. Bei maximaler Blende `D = 48 mm` wächst das Nah-Bündel auf **20,16 mm**. Die Ansichtsbuttons wechseln die Darstellung unter Beibehaltung der Szene.

## Bedienung

- Szenentiefe und Objekthöhe per Regler oder durch Ziehen des blauen Szenenpunkts.
- **Mausrad über dem Strahlengang:** 1–12-facher Zoom um die Mausposition. **Hintergrund ziehen:** vergrößerten Ausschnitt verschieben. **Ansicht zurücksetzen:** wieder den gesamten Strahlengang zeigen. Die Objekt- und Strahlauswahl funktioniert auch im vergrößerten Ausschnitt. Zoom und Ausschnitt werden in Zustandslinks und HTML-Export gespeichert.
- Blendendurchmesser 8–48 mm; 5–25 visualisierte Strahlen. Die Strahlzahl verändert nur die Darstellung, nicht die kontinuierliche Geometrie.
- Strahlposition `v` innerhalb der aktuellen Blende; als URL-Zustand `sample = v/(D/2)`.
- `α` zwischen 0,8 und 1,8 in Ansicht 3. Damit ist der gesamte einstellbare Tiefenbereich refokussierbar. Das Ray-Diagramm bleibt immer auf die ursprüngliche Sensorebene `b` bezogen.
- Gewähltes Integrationspixel `y` bzw. `y′` zwischen −40 und +40 mm. Die Linienschar kann ausgeblendet werden; der ausgewählte Integrationspfad bleibt sichtbar.
- **1 / 2 / 3:** Ansichten; **← / →:** Szenentiefe in Schritten von 5 mm; **R:** Reset. Fokussierte Bedienelemente behalten ihre native Tastaturbedienung.
- **Zustandslink kopieren:** Vollständige Konfiguration teilen; Clipboard-Fallback mit auswählbarem Linkfeld.
- **Widget mit Zustand speichern:** Eigenständige HTML-Datei herunterladen. URL-Parameter überschreiben den eingebetteten Startzustand.

## Optisches Modell

Paraxiale dünne Linse mit `f = bg₀/(b+g₀) ≈ 69,23 mm`, ursprünglicher Sensor bei `b = 90 mm`. Die zugehörige Fokusebene bleibt `g₀ = 300 mm`. Für einen Szenenpunkt `(g,Y)` gilt:

```text
b★ = fg/(g−f)
y(v) = (1−b/b★)v − bY/g
y′(v) = (1−b′/b★)v − b′Y/g
c_b = D |1−b/b★|
c_b′ = D |1−b′/b★|
```

`c` ist die Bündelbreite im gezeigten Schnitt (bei kreisförmiger Blende der geometrische Zerstreuungskreisdurchmesser). Das Widget zeigt einen Schnitt durch das 4D-Lichtfeld bei festen `x,u` und verwendet deshalb 1D-Integrale:

```text
g_b(y) = ∫ L_b(y,v) dv
g_b′(y′) = ∫ L_b((1−1/α)v + y′/α, v) dv
y′ = αy + (1−α)v
```

Die Integrationsgrenzen sind `−D/2 … +D/2`. Im 4D-Modell wird zusätzlich über `u` integriert. Die Linien visualisieren das Integral geometrisch; es wird kein radiometrisches Bild synthetisiert.

**Vorzeichen:** Horizontal `y`, vertikal `v`. Als Scherung ist `dy/dv` angegeben (nicht die Zeichensteigung `dv/dy`). Im Fokus ist sie 0: die Lichtfeldgerade steht senkrecht. Vor der Fokusebene ist sie positiv, dahinter negativ. Die Integrationsscherung ist `1−1/α`. Bei `α★ = b★/b` stimmen beide überein.

Der Strahlengang ist schematisch, mit unterschiedlichen axialen Maßstäben auf Objekt- und Bildseite. Das Ray-Diagramm hat feste Achsen in mm. Beugung, Aberrationen und radiometrische Faktoren werden nicht modelliert.

## Einbettung und Links

```html
<iframe
  src="widgets/3.2%20Lichtfeld-Ray-Diagramm/index.html?lang=de"
  title="Kamera-Lichtfeld und Ray-Diagramm"
  width="100%"
  height="1800"
  style="border:0"
  loading="lazy">
</iframe>
```

URL-Parameter:

| Parameter | Werte / Bedeutung |
|---|---|
| `mode` | `single`, `integrate`, `refocus` |
| `depth` | 125–700 mm |
| `height` | −30…30 mm |
| `aperture` | 8–48 mm |
| `sample` | −1…1; normierte Blendenposition |
| `rays` | 5–25, ungerade |
| `alpha` | 0.8–1.8; wirksam in `refocus` |
| `pixel` | −40…40 mm |
| `family` | `true`, `false` |
| `lang` | `de`, `en` |
| `dark` | `true`, `false` |
| `zoom` | 1–12; Vergrößerung des Strahlengangs |
| `viewX`, `viewY` | Ausschnittmittelpunkt in SVG-Koordinaten (0–760 bzw. 0–440); auf die Zeichnungsgrenzen begrenzt |

Beispiele für die Vorlesung:

```text
index.html?lang=de&mode=single
index.html?lang=de&mode=integrate&depth=300&pixel=6
index.html?lang=de&mode=refocus&depth=125&alpha=1&pixel=14.4&aperture=48
index.html?lang=de&mode=refocus&depth=125&alpha=1.7241379310344827&pixel=24.82758620689655&aperture=48
index.html?lang=de&mode=refocus&depth=600&alpha=0.8695652173913043&pixel=2.608695652173913
```

Lokale `file:`-Links gelten auf dem aktuellen Gerät. Für geräteübergreifendes Teilen die HTML-Datei weitergeben oder unter einer gemeinsamen Web-Adresse bereitstellen. Screenshot-Selektoren: `#camera`, `#rayDiagram`, `.views`.

## Browserprüfung

Mit installiertem Playwright und Chromium:

```text
python "widgets/3.2 Lichtfeld-Ray-Diagramm/verify_browser.py"
```

Prüft die projizierten Strahlkoordinaten gegen eine unabhängige paraxiale Berechnung für 15 Szenen, die Deckung von Lichtfeld- und Integrationsgerade nach Refokussierung, Presets, Ziehen des Szenenpunkts, Sprach-/Themewechsel, Wiederöffnen des HTML-Exports, mobile Überlaufgrenzen und ungültige URL-Parameter. Desktop- und Mobil-Screenshots werden im temporären OpenCode-Verzeichnis abgelegt.
