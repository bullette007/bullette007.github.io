# 2.10 · Wie eine Fresnel-Linse entsteht

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget ohne externe Bibliotheken. Deutsch/Englisch, Hell/Dunkel, responsive Darstellung. Standard: Englisch und dunkles Farbschema; `?lang=de` startet auf Deutsch.

## Die Animation

Ein geometrisch maßstäblicher Querschnitt zeigt die Konstruktion einer refraktiven Fresnel-Linse.

1. **Konvexe Linse:** Eine rotationssymmetrische plankonvexe Linse mit paraboloidischer Oberfläche.
2. **Ringe schneiden:** Zylindrische Schnitte teilen die Linse in eine zentrale Scheibe und konzentrische Ringzonen. Die gekrümmten oberen Kappen bleiben erhalten.
3. **Ballast entfernen:** Das graue Material unterhalb dieser Kappen verschwindet. Die Kappen befinden sich noch an ihren ursprünglichen Positionen; die dünne Grundplatte bleibt bestehen.
4. **Flach stapeln:** Die Kappen werden parallel zur optischen Achse auf die gemeinsame Grundplatte abgesenkt. Sie bleiben radial nebeneinander. Ihre lokale Neigung und Krümmung ändern sich nicht.

Die Animation lässt sich abspielen, pausieren oder mit dem Konstruktionsregler vorwärts und rückwärts durchfahren. Die vier Schaltflächen springen zu den Schlüsselzuständen bei 0, 25, 50 und 100 Prozent.

## Bedienung

- **Zonen:** 3–16 konzentrische Bereiche einschließlich zentraler Scheibe; gleiche radiale Breiten.
- **Wölbung / Radius:** 0,2–0,7; verändert die Ausgangsdicke und Flächenneigungen.
- **Animationsdauer:** 5–20 Sekunden für die vollständige Konstruktion.
- **Ring untersuchen:** Ein ausgewählter Bereich ist orange. Auswahl per Regler oder Klick in den Querschnitt.
- **Ursprüngliche Kontur:** Gestrichelte Referenz im Querschnitt.
- **Flächennormale vergleichen:** Pfeile am ursprünglichen und verschobenen Flächenstück machen die unveränderte Neigung sichtbar.
- Kennzahlen zeigen ursprüngliche Dicke, maximale Enddicke und eingespartes Volumen einschließlich Grundplatte.
- **Leertaste:** Abspielen/Pause; **←/→:** Fortschritt um zwei Prozentpunkte verändern; **R:** Reset. Fokussierte Bedienelemente behalten ihre native Tastaturbedienung.
- **Zustandslink kopieren:** Vollständige Konfiguration teilen; Clipboard-Fallback mit auswählbarem Linkfeld.
- **Widget mit Zustand speichern:** Vollständige HTML-Datei mit aktueller Konfiguration als Startzustand herunterladen. URL-Parameter überschreiben den eingebetteten Zustand.

Lokale `file:`-Links gelten auf dem aktuellen Gerät. Für geräteübergreifendes Teilen die HTML-Datei weitergeben oder unter einer gemeinsamen Web-Adresse bereitstellen.

## Geometrie und optische Einordnung

Radius R = 1, Grundplattendicke b = 0,06 R, Wölbung s. Ausgangsprofil:

```text
t(r) = b + s(1 − r²),   0 ≤ r ≤ 1
```

Für eine Zone mit innerem Radius a und äußerem Radius c wird eine Kappe oberhalb der Ebene `b + s(1−c²)` behalten und um `s(1−c²)` abgesenkt. Endprofil innerhalb der Zone:

```text
tF(r) = b + s(c² − r²)
dtF/dr = dt/dr = −2sr
```

Die vollständigen gekrümmten Flächenstücke bleiben erhalten; sie werden nicht durch gerade Prismen ersetzt. Der Querschnitt hat identische Maßstäbe für Radius und Höhe und wird auf Canvas gezeichnet.

Volumen einschließlich Grundplatte:

```text
Voriginal = πb + πs/2
VFresnel  = πb + (πs/2) Σ (c² − a²)²
```

Gleiche Flächennormale erklärt das refraktive Prinzip. Da die Grenzflächenpositionen verändert werden, sind identischer Fokus und identische optische Weglänge bei endlicher Dicke dadurch nicht exakt garantiert. Das Widget zeigt die geometrische Konstruktion, keine Strahlverfolgung oder optimierte optische Auslegung. Reale Linsen werden typischerweise geformt, geprägt oder spanend hergestellt. Eine refraktive Fresnel-Linse ist von einer beugungsbasierten Fresnel-Zonenplatte zu unterscheiden.

## Einbettung

```html
<iframe
  src="widgets/2.10%20Fresnel-Linse/index.html?lang=de"
  title="Wie eine Fresnel-Linse entsteht"
  width="100%"
  height="1650"
  style="border:0"
  loading="lazy">
</iframe>
```

URL-Parameter: `progress=0…100`, `zones=3…16`, `sag=0.2…0.7`, `duration=5…20`, `inspect=1…zones`, `ghost=true|false`, `normals=true|false`, `lang=de|en`, `dark=true|false`. Die Oberfläche aktualisiert die URL bei manuellen Änderungen und beim Ende/Pausieren der Animation; sie startet beim Öffnen nicht automatisch.

Endzustand für die Vorlesung:

```text
index.html?lang=de&progress=100&zones=7&sag=0.5&inspect=4
```

Screenshot-Selektoren: `#section`, `.views`.
