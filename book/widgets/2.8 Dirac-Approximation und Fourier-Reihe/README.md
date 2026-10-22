# 2.8 · Dirac-Approximation & Fourier-Reihe

`index.html` direkt im Browser öffnen. Das Widget läuft ohne Server, Internet oder externe Bibliotheken und bietet eine deutsche und eine englische Oberfläche.

## Experimente

- **Dirac-Approximation:** Rechteckimpuls aus `plotDiracEps(eps)` mit ε zwischen 0,01 und 2. Höhe 1/ε, Fläche exakt 1. Umschaltbare feste Höhenskala und Vergleich der Testfunktion φ(x) mit ihrem Mittelwert unter dem Impuls veranschaulichen die Abtasteigenschaft.
- **Fourier-Reihe:** Trigonometrische und komplexe Darstellung der Beispiele aus `reconstrTrigFourier(terms)` und `reconstrExpFourier(terms)`. Sägezahn mit vier Perioden im Intervall L = 20; das komplexe Skript-Beispiel mit Offset +3 ist ein eigenes Preset. Dazu Rechteck, Dreieck, Sinus und frei zeichnbare Signale.
- **Analyse und Synthese:** Signale mit Maus, Stift oder Finger zeichnen. Signierte Balken für a_f und b_f bzw. zweiseitige Magnituden |c_f|. Harmonische über Balken oder Tastaturschaltflächen einzeln aktivieren, deaktivieren oder isoliert als Einzelbeitrag untersuchen. RMSE und erfasste Signalenergie aktualisieren sich live.
- **Koeffizienten direkt bearbeiten:** Balken vertikal ziehen; links im Frequenzpaar a_f (violett), rechts b_f (orange). Auch Nullkoeffizienten lassen sich an der Nulllinie greifen. Die grüne Synthese und der orange Einzelbeitrag reagieren live, das blaue Zielsignal bleibt erhalten. In komplexer Darstellung ändert Ziehen den Betrag bei erhaltener Phase; bei einem Nullkoeffizienten startet die Phase bei 0. Beide konjugierten Frequenzen ändern sich gemeinsam.

## Bedienung

- Regler: Pfeiltasten bei fokussiertem Regler.
- Register: Links/Rechts, Home/End bei fokussiertem Register.
- `R`: aktuelles Experiment zurücksetzen; Sprache und Farbschema bleiben erhalten.
- `Strg/Cmd+Z`: letzte Zeichnung bzw. Signaländerung rückgängig machen (eine Stufe).
- Spektrum: vertikal ziehen zum Bearbeiten, kurz klicken zum Ein-/Ausschalten. Das Ziehen aktiviert die bearbeitete Harmonische und nimmt sie bei Bedarf in H auf. Während des Ziehens bleibt die Spektralskala fest; danach passt sie sich an. Maus, Stift und Touch werden unterstützt.
- **Koeffizienten aus Zielsignal:** manuelle Änderungen zurücknehmen und das blaue Zielsignal erneut analysieren. Zeichnen oder ein neues Signal-Preset ersetzt ebenfalls die manuellen Koeffizienten. Rückgängig stellt auch Koeffizientenänderungen wieder her.
- **Zustandslink kopieren:** Einstellungen, deaktivierte Harmonische und eigene Zeichnung teilen. Bei lokalen Dateien verweist der Link auf den lokalen Dateipfad; zum Teilen zwischen Geräten die Datei unter einer gemeinsamen Web-Adresse bereitstellen.
- **Widget mit Zustand speichern:** Eine vollständige HTML-Datei mit dem aktuellen Zustand als Startkonfiguration herunterladen. Diese Datei lässt sich auch offline weitergeben.

## Numerische Konventionen

Wie im Skript gilt `g(x) = a₀ + Σ [a_f cos(2πfx/L) + b_f sin(2πfx/L)]`: a₀ ist der Mittelwert. `c₀ = a₀`, `c_f = (a_f − j b_f)/2`, `c_−f = c_f*`. Komplex konjugierte Frequenzpaare werden gemeinsam geschaltet, damit die Rekonstruktion reell bleibt.

Der Regler **H** gibt den höchsten Frequenzindex an, nicht die Anzahl der Summanden. Bei allen aktivierten Beiträgen sind es H Harmonische plus DC bzw. 2H+1 komplexe Terme. Damit ist die Obergrenze gegenüber den unterschiedlich gezählten Notebook-Reglern eindeutig.

Analyse: 1024 äquidistante Intervallmittelpunkte, periodische Quadratur, höchstens H = 100. Zeichnungen sind Funktionsgraphen, keine zweidimensionalen Bilder; Wertebereich −1,5 bis 4,5. Beim Zeichnen über mehrere angezeigte Perioden wird stets dieselbe Grundperiode bearbeitet. Die URL-Zeichnung verwendet 16-Bit-Werte mit Schrittweite 1/6000; maximaler Quantisierungsfehler 1/12000.

Die Fläche des Rechteckimpulses wird analytisch angegeben. Auch das Testfunktionsintegral wird analytisch berechnet: `1 + 0,5 sin(ε/2)/(ε/2)`. Dadurch bleibt die Aussage auch bei sehr kleinen ε unabhängig von einem Plot-Abtastraster korrekt.

Manuell bearbeitete Koeffizienten werden getrennt vom Zielsignal gespeichert und bleiben beim Wechsel zwischen trigonometrischer und komplexer Ansicht erhalten. Die Energieanzeige zeigt dann das Verhältnis Syntheseenergie/Zielsignalenergie (auch über 100 % möglich), statt dieses als erfasste Zielsignalenergie zu interpretieren. Die Zeitachse passt ihren Wertebereich bei Bedarf an. Koeffizienten sind auf Beträge bis 100 begrenzt; Änderungen stehen als `edits` im Zustandslink und im HTML-Export.

## Einbettung

```html
<iframe
  src="widgets/2.8%20Dirac-Approximation%20und%20Fourier-Reihe/index.html?tab=fourier&representation=exp&preset=offset&terms=32"
  title="Dirac-Approximation und Fourier-Reihe"
  width="100%"
  height="1250"
  style="border:0"
  loading="lazy">
</iframe>
```

Weitere URL-Parameter: `epsilon`, `fixedScale`, `frequency`, `inspect`, `showComponent`, `repeat`, `lang=de|en`, `dark=true|false`, `off` (kommaseparierte deaktivierte Frequenzindizes). Die Oberfläche erzeugt vollständige Zustandslinks automatisch.
