# 2.4 · Faltungs-Scrubber

`index.html` direkt im Browser öffnen. Das Widget ist eine eigenständige HTML-Datei ohne externe Abhängigkeiten und funktioniert offline.

## 1D-Modus

- **f und h wählen:** Rechteck, Dreieck, Gauß, einseitiger Exponentialpuls, bipolarer Puls, Einheitsimpuls oder eigene Zeichnung. Die Breiten sind unabhängig einstellbar.
- **Zeichnen:** In den oberen Signaldiagrammen mit Maus oder Finger ziehen. Die bestehende Kurve wird zur editierbaren Zeichnung; „f leeren“ / „h leeren“ setzen sie auf null. Auch negative Werte sind möglich.
- **Ausgabeposition x:** Der Regler verschiebt den gespiegelten Kernel `h(x − α)`. Die Integrationsvariable heißt wie im Skript **α**.
- **Produkt:** Die tatsächlich integrierte Produktfläche ist schraffiert. Positive und negative Beiträge sind orange bzw. rot.
- **Ergebnis:** Baut sich von links bis zur aktuellen Position auf. Durch Klicken im Ergebnisdiagramm eine Position auswählen; optional das vollständige Ergebnis anzeigen.
- **Presets:** `Rect ∗ Rect`, `f ∗ δ = f` und `δ ∗ h = h` für das Impulsantwort-Argument.

Die Rechnung nutzt 241 Stützstellen auf `[−6, 6]` und `Δα = 0,05`. Außerhalb dieses Intervalls sind die Signale null. Das lineare Faltungsergebnis hat 481 Stützstellen auf `[−12, 12]`. Der Gauß- und Exponentialpuls werden am Rand dieses Bereichs abgeschnitten. Rechteck-Grenzpunkte tragen den halben Wert. Ein Impuls wird numerisch durch eine Stützstelle der Höhe `1/Δα` repräsentiert (Fläche 1), grafisch durch einen symbolischen Pfeil. Die Delta-Presets reproduzieren deshalb die abgetastete Originalfunktion exakt.

## 2D-Modus

12 × 12-Testbild, editierbarer 3 × 3-Kernel und rasterweise aufgebaute Ausgabe. Presets: normierter Gauß, Mittelwert, Identität, asymmetrischer Kernel und Sobel x. Einzelimpuls und Schachbrett sind weitere Testbilder.

Die vergrößerte Kachel zeigt Nachbarschaft, **um 180° gedrehten Kernel**, neun Produkte und deren Summe. Es wird eine echte Faltung mit Zero-Padding und gleich großer Ausgabe berechnet, keine Kreuzkorrelation. `(m,n)` bezeichnet `(Zeile, Spalte)`, nullbasiert. Kernelwerte sind zwischen −20 und 20 frei editierbar; eigene Kernel werden nicht automatisch normiert. Negative Ergebnisse erscheinen rot. Die Ausgabe-Farbskala bleibt während eines Durchlaufs konstant; ihr Betrag wird unter dem Bild angegeben.

## Bedienung und Teilen

- **← / →:** ein Schritt; **Pos1 / Ende:** Anfang / Ende; **Leertaste:** Abspielen / Pause; **R:** Reset. Bei fokussierten Bedienelementen gilt deren native Tastaturbedienung.
- **Tempo:** 10–100 Stützstellen bzw. Pixel pro Sekunde. Am Ende hält die Animation an.
- **Hell / Dunkel** und responsives Layout für Beamer, Tablet und Mobilgerät.
- **Deutsch / English:** Englisch ist die Standardsprache. Der Umschalter übersetzt beide Modi einschließlich Hilfetexten und Bedienbeschreibungen. Mit `?lang=de` direkt auf Deutsch starten; `?lang=en` wählt Englisch. Die Sprache wird im Link gespeichert.
- **Link kopieren:** Modus, Signale, Breiten, Position, Tempo, Farbschema, vollständige Ausgabe sowie aktive Zeichnungen und eigene Kernel stehen in der URL. Ohne Clipboard-Zugriff erscheint ein auswählbares Linkfeld.
- **Reset** stellt die Anfangswerte wieder her und behält Modus, Farbschema und Sprache bei.

Beispiele:

```text
index.html?f=rect&h=rect&fw=1&hw=1&x=0&full=1
index.html?f=bipolar&h=delta&x=0.5
index.html?mode=2d&imageType=impulse&kernelType=asymmetric&pixel=78&full=1
```

## Einbettung im Web-Buch

In einer Raw-HTML-Zelle, Pfad relativ zur Buchseite anpassen:

```html
<iframe
  src="widgets/2.4%20Faltungs-Scrubber/index.html"
  title="Interaktiver Faltungs-Scrubber"
  style="width:100%;height:1150px;border:0"
  loading="lazy">
</iframe>
```

Für die Vorlesung die HTML-Datei direkt im Browser öffnen und Browser-Vollbild nutzen. `preview.png` zeigt den Standardzustand für eine statische Einbettung; ein veröffentlichter Kurzlink / QR-Code kann auf dieselbe HTML-Datei zeigen.

## Kurze Aktivierung

1. **Vorhersagen:** „Was entsteht aus zwei Rechtecken? Wo beginnt und endet das Ergebnis?“
2. **Beobachten:** `Rect ∗ Rect` auswählen und den Regler / die Animation laufen lassen.
3. **Erklären:** „Warum entspricht die Höhe des Dreiecks hier der Überlappungslänge? Warum reicht dieses Argument beim bipolaren Puls nicht mehr?“

Anschluss: „Warum liefert ein Einheitsimpuls genau die Impulsantwort?“ Dazu `δ ∗ h = h` wählen. Im 2D-Modus den asymmetrischen Kernel benutzen und vor dem Aufdecken der Produkte die Richtung seiner Wirkung vorhersagen lassen.

## Verifikation

```powershell
python "widgets/2.4 Faltungs-Scrubber/verify_browser.py"
```

Benötigt für die Entwicklung Python mit Playwright und installiertem Chromium. Prüft Offline-Aufruf, Referenzfaltungen, Impuls-Presets, Spiegelung, Randbehandlung, Zeichnen, URL-Wiederherstellung, Animation, Tastaturbedienung und mobile Darstellung. Die Prüfdatei ist keine Laufzeitabhängigkeit des Widgets.
