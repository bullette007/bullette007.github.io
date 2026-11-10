# 2.9 · DFT & zyklische Fortsetzung

`index.html` direkt im Browser öffnen. Eigenständiges HTML-Widget ohne Server, Bibliotheken oder Internetzugriff. Deutsch/Englisch, Hell/Dunkel, responsive Darstellung. Standard: Englisch und dunkles Farbschema; `?lang=de` startet auf Deutsch.

## 2D · Bild, Faltung & Padding

Oben auf **2D · Bild, Faltung & Padding** wechseln oder direkt `index.html?mode=2d&lang=de` öffnen.

- Ein **16 × 16-Pixel-Bild** (Häuschen am Rand, Randmarken oder Eckimpuls) wird als 3 × 3-Kachelung angezeigt. Blau umrahmt das ursprüngliche Bild, violett die gepaddete DFT-Periode. Schwarz entspricht null.
- Ein verschiebbarer Kernel markiert seinen Fußabdruck orange. Aktive Zugriffe außerhalb der zentralen Periode werden rot hervorgehoben. Klick auf eine Bildkopie oder ein Ergebnisbild bzw. die x/y-Regler wählt das Ausgabepixel.
- Mittelwert-, endlicher Gauß- und horizontaler Bewegungskernel, jeweils mit ungerader Kantenlänge K = 3…11. Alle Kernel sind auf Summe 1 normiert. Die Kernelvorschau skaliert auf ihren größten Gewichtswert.
- **Nullrand p pro Seite** ist von 0 bis 6 Pixeln einstellbar. „Ohne Padding“ und „Genügend Padding“ springen auf p = 0 bzw. p = (K−1)/2.
- Drei Ergebnisbilder vergleichen die auf 16 × 16 beschnittene zyklische DFT-Faltung, die direkt berechnete lineare Faltung mit Nullfortsetzung und ihren absoluten Unterschied. Feste Helligkeitsskala [0,1]; Fehlerbild rot mit vierfacher Verstärkung. Der maximale tatsächliche Fehler und die Zahl abweichender Pixel stehen darüber (Toleranz 10⁻¹⁰ für den Zähler).

### Empfohlener Versuch

1. Häuschen und 7 × 7-Mittelwertkernel wählen; p = 0. Den Kernel rechts am Bildrand platzieren: Er erreicht das Häuschen der nächsten Kopie. Im zyklischen Ergebnis erscheint rechts ein heller Schleier, den die lineare Referenz nicht besitzt.
2. Padding schrittweise erhöhen. Die Kopien rücken auseinander. Ab **p = 3 pro Seite** verschwinden die Fehler im Originalfenster.
3. Kernel auf 11 × 11 vergrößern: Nun sind **p = 5 pro Seite** ausreichend. Über „Genügend Padding“ direkt vergleichen.

### Warum „eine halbe Kernelbreite“?

Für einen zentrierten Kernel mit ungerader Kantenlänge K ist sein Radius r = (K−1)/2. **p ≥ r pro Seite** garantiert die Übereinstimmung im gesamten ursprünglichen Bildfenster, unabhängig vom Bildinhalt. Das ergibt eine DFT-Kantenlänge M = 16 + 2p ≥ 16 + K−1, ausreichend auch für die vollständige lineare Faltung ohne Überlagerung. Der Nullrand ist damit insgesamt mindestens K−1 Pixel breit, verteilt auf beide Seiten. Die zyklische Randbedingung bleibt bestehen, stört aber den behaltenen Ausschnitt nicht mehr. Bei kleineren Rändern kann ein konkretes Bild stellenweise oder vollständig fehlerfrei sein; die allgemeine Garantie gilt erst ab der genannten Breite.

Die zyklische Faltung wird tatsächlich durch separable 2D-DFT, komplexe Spektrenmultiplikation und inverse 2D-DFT berechnet, auch bei nicht-zweierpotenzgroßen Perioden. Der Kernel wird zentriert um DFT-Index (0,0) eingebettet. Die lineare Referenz wird unabhängig mit räumlichen Summen und Nullfortsetzung berechnet. Für das Ausgabepixel (x,y) wird g(x−kx,y−ky) h(kx,ky) summiert: echte Faltung, keine Kreuzkorrelation.

## 1D · Vier gekoppelte Schritte

1. **Kontinuierlich:** Ausgangssignal und kontinuierliche Fourier-Transformierte.
2. **Im Ort abtasten:** Abtastabstand Δx erzeugt ein periodisches Spektrum mit Periode fA = 1/Δx. Ein Aliasing-Preset zeigt einen Sinus oberhalb der Nyquist-Grenze.
3. **Fenster auswählen:** N Samples bleiben erhalten. Ihre Transformierte S(f) ist weiterhin eine kontinuierliche Funktion von f, periodisch mit fA. Rechteck oder Hann wählbar.
4. **Frequenzen abtasten:** DFT-Werte mit Δf = 1/L erscheinen gleichzeitig mit zyklischen Ortskopien im Abstand L. Ein Regler oder eine Animation blendet beides gemeinsam ein. Dies verändert nur die Deckkraft, nicht die zugrunde liegende Mathematik.

Der Cursor lässt sich über die Blockgrenze bewegen: Nach M−1 kommt M, also wieder Sample 0. Der zugehörige Wert im ursprünglichen Block wird gleichzeitig markiert. **Zyklisch bedeutet g̃[M] = g̃[0], nicht g[M−1] = g[0].** Ohne Padding ist M = N.

Die zuschaltbare **Synthese-Lupe** zeigt die vollständige periodische IDFT-Interpolation und eine normierte Cosinus-Basiskomponente. Jede Basis ist M-periodisch, daher auch die Summe. Der Basisindex lässt sich mit einem Regler auswählen.

## Drei Vorlesungsversuche

- **Woher kommt die Periodizität?** Mit dem Exponentialsignal die Schritte 1–3 durchgehen. In Schritt 3 ist die Frequenzvariable noch kontinuierlich. Schritt 4 wählen, Einblendregler auf 0 setzen und animieren. Frequenzabtastung und räumliche Kopien entstehen gemeinsam. Mit dem Cursor über M−1 hinauswandern.
- **Leakage:** „Sinus: 3 Perioden“ mit „Sinus: 3,6 Perioden“ vergleichen. Die rechteckgefensterte Spektralkurve besitzt auch beim ganzzahligen Beispiel Nebenkeulen, aber die übrigen DFT-Stellen liegen dort auf Nullstellen. Beim nichtganzzahligen Beispiel fällt Energie auf viele Bins. Hann reduziert Nebenkeulen bei breiterer Hauptkeule. Ein Randsprung ist anschaulich, aber kein allgemeines notwendiges Kriterium für Leakage.
- **Zero-Padding:** Bei gleichem Signal von ×1 auf ×2 oder ×4 wechseln. Die Spektralkurve bleibt gleich, das Frequenzraster wird dichter, und die räumliche Wiederholungsperiode wächst. Das erzeugt keine neue Information und verbessert nicht die Trennfähigkeit benachbarter Frequenzen.

## Bedienung und Teilen

- Schritte über die vier Schaltflächen oder **← / →** wechseln, solange kein Bedienelement fokussiert ist.
- Regler unterstützen ihre native Pfeiltastenbedienung; **R** setzt zurück und erhält Sprache/Farbschema.
- Cursor per Regler, −1/+1, automatischem Lauf oder Klick im Ortsplot bedienen.
- Die komplette Einstellung steht in der URL. Bei gesperrtem Clipboard erscheint ein auswählbares Linkfeld.
- **Widget mit Zustand speichern** lädt eine vollständige Offline-HTML-Datei mit dem aktuellen Startzustand herunter. URL-Parameter überschreiben den eingebetteten Zustand.
- Lokale `file:`-Links beziehen sich auf das aktuelle Gerät. Zum geräteübergreifenden Teilen HTML-Datei versenden oder unter einer gemeinsamen Web-Adresse bereitstellen.

## Numerische Konventionen

- Fourier-Kern exp(−i2πfx). Ortsabtastkamm mit Δx gewichtet.
- Endliche Folge: y[n] = g(nΔx)w[n] für n = 0,…,N−1; außerhalb null. `S(f) = Δx Σ y[n] exp(−i2πfnΔx)`.
- Unnormierte DFT `X[k] = Σ y[n] exp(−i2πkn/M)`, inverse DFT mit Faktor 1/M. Das Frequenzdiagramm zeigt `|S(f)|`, somit an den DFT-Stellen `Δx |X[k]|`. Alle Summen werden direkt berechnet.
- M = N × Paddingfaktor, L₀ = NΔx, L = MΔx, Δf = 1/L. DFT-Punkte werden auch außerhalb des zentralen Nyquist-Intervalls periodisch dargestellt.
- Exponentialsignal: exp(−x/2) für x ≥ 0, sonst null. g[0] = 1. Kontinuierliche FT analytisch; unendliche abgetastete Folge in Schritt 2 als exakte geometrische Summe. Die Wahl des Einzelwertes am Sprung beeinflusst die kontinuierliche FT nicht, die Abtastwerte aber schon.
- Sinus: sin(2πf₀x + 0,35) mit f₀ = cycles/(NΔx). In Schritten 1/2 werden die Dirac-Spektrallinien symbolisch dargestellt. Ihre Pfeilhöhen sind keine endlichen Funktionswerte. Das Aliasing-Preset verwendet 6,3 Perioden bei N = 8 und vermeidet entartete zusammenfallende Linien.
- Hann: `(1−cos(2πn/(N−1)))/2`; keine Amplitudenkompensation. Rechteckfenstergrenzen: −Δx/2 und (N−1/2)Δx.
- Gestrichelte Ortsverbindungen sind Lesehilfen für diskrete Werte. Die Synthese-Lupe verwendet zwischen ganzzahligen n vorzeichenbehaftete Frequenzindizes und den Nyquist-Cosinus. Dies ist eine mögliche reelle periodische Interpolation; an den Samples stimmt sie mit der IDFT überein.

## Einbettung

```html
<iframe
  src="widgets/2.9%20DFT%20und%20zyklische%20Fortsetzung/index.html?lang=de&amp;step=4"
  title="DFT und zyklische Fortsetzung"
  width="100%"
  height="1600"
  style="border:0"
  loading="lazy">
</iframe>
```

URL-Parameter: `step=1…4`, `preset=decay|sine`, `count=8…64`, `dx=0.1…1`, `cycles=0.5…12`, `window=rect|hann`, `padding=1|2|4`, `reveal=0…100`, `cursor`, `harmonic`, `showBasis=true|false`, `lang=de|en`, `dark=true|false`. Die Oberfläche erzeugt vollständige Links. Screenshot-Selektoren: `#space`, `#frequency`, `#basis`, `.views`.

Für 2D zusätzlich: `mode=2d` (Standard `1d`), `image2d=house|blocks|impulse`, `filter2d=box|gauss|motion`, `size2d=3|5|7|9|11`, `pad2d=0…6`, `pixelX=0…15`, `pixelY=0…15`. Screenshot-Selektoren: `#tiles2d`, `#kernel2d`, `#cyclic2d`, `#linear2d`, `#error2d`. Link und HTML-Export enthalten beide Modi. Reset setzt den aktuellen Modus zurück; Sprache, Farbschema und Modus bleiben erhalten.

Direktlink für die Demo ohne Padding:

```text
index.html?mode=2d&lang=de&image2d=house&filter2d=box&size2d=7&pad2d=0&pixelX=15&pixelY=8
```
