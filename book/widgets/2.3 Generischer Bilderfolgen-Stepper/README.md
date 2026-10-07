# 2.3 · Generischer Bilderfolgen-Stepper

`index.html` direkt im Browser öffnen. Die Datei enthält eine fünfteilige SVG-Demo und benötigt weder Installation noch Internetverbindung oder externe Bibliotheken.

## Bedienung

- **Zurück / Weiter**, Pfeiltasten oder Schieberegler: Bild auswählen.
- **Pos1 / Ende**: erster / letzter Schritt; **R / Reset**: erster Schritt.
- **Crossfade**: sanft überblenden (bei reduzierter Bewegung deaktiviert).
- **Hell / Dunkel**: Darstellung umschalten.
- **Link kopieren**: Folge und Zustand teilen. Schritt, Crossfade und Farbschema stehen in der URL.
- Unter **Eigene Bilderfolge laden**: Bildadressen einfügen oder lokale SVG-/PNG-Dateien auswählen. Lokale Dateien werden numerisch nach Dateiname sortiert, bleiben nur während der Browsersitzung verfügbar und sind nicht per Link teilbar.

## Eigene Folgen einbinden

### Nummerierte Dateien über URL-Parameter

```text
index.html?prefix=../../figures/imaging_pinhole_lens_&start=0&end=5&ext=svg
```

`start` und `end` sind inklusive. Das Beispiel lädt sechs Bilder von `imaging_pinhole_lens_0.svg` bis `imaging_pinhole_lens_5.svg`. Passen Sie die Grenzen an die vorhandenen Dateien an. Relative Pfade beziehen sich auf **die HTML-Datei**, nicht auf die einbettende Buchseite.

| Parameter | Bedeutung | Standard |
| --- | --- | --- |
| `prefix` | Pfad einschließlich Dateinamen-Präfix | eingebaute Demo |
| `start`, `end` | erste / letzte Dateinummer | `start=0`; `end` erforderlich |
| `ext` | `svg` oder `png` | `svg` |
| `pad` | Mindestbreite der Dateinummer, etwa `3` für `007` | `0` |
| `step` | angezeigter Schritt, **1-basiert**, unabhängig von `start` | `1` |
| `fade` | `1` = Crossfade, `0` = direkt umschalten | `0` |
| `theme` | `light` oder `dark` | `light` |
| `title` | Titel des Widgets | Bilderfolgen-Stepper |
| `ui` | `minimal` = nur Bild, Zurück/Weiter, Schrittzähler und Schieberegler; `full` = vollständige Oberfläche | `full` |
| `resize` | `auto` = Höhe aus Bildformat und verfügbarer Breite berechnen und an die Elternseite melden | feste iframe-Höhe |

Werte mit Leerzeichen, `&`, `#` oder anderen URL-Sonderzeichen müssen URL-kodiert sein. Beispiel: `title=Linsenprinzip&step=3&fade=0&theme=dark`. Bis zu 10.000 nummerierte Bilder sind möglich; nur das ausgewählte Bild wird geladen.

### Reduzierte Oberfläche im Notebook

Mit `ui=minimal` werden Kopfzeile samt Farbschema-Schalter, Bildunterschrift, zusätzliche Fortschrittsleiste, Reset-/Crossfade-/Link-Optionen, Status, Tastaturhinweis und Dateiauswahl ausgeblendet. Tastatursteuerung und die Einstellungen `fade` und `theme` funktionieren weiterhin. Bild-Ladefehler bleiben direkt im Bildbereich sichtbar.

Bei laufendem Webserver auf Port 8000 diese Code-Zelle verwenden:

```python
from IPython.display import IFrame, display

display(IFrame(
    src="http://127.0.0.1:8000/widgets/"
        "2.3%20Generischer%20Bilderfolgen-Stepper/index.html"
        "?ui=minimal&step=1&fade=0&theme=light"
        "&prefix=../../figures/4/desk_lightsources_example_"
        "&start=1&end=5&ext=svg",
    width="100%",
    height=740,
))
```

Ohne `ui=minimal` (oder mit `ui=full`) erscheint die vollständige Oberfläche. Im Minimalmodus füllt die Bildkarte die iframe-Höhe aus; der Bildbereich nutzt den Platz oberhalb der Navigation. Die iframe-Höhe bestimmt damit die Größe der Darstellung (z. B. `height=600` für eine kompaktere Ausgabe oder `height=740` für größere Bilder).

### Automatische iframe-Größe in Notebook, RISE und Jupyter Book

Die Hilfsfunktion `widgets/stepper.py` erzeugt ein iframe mit 100 % Breite und einem Listener für Höhenmeldungen. Der Stepper berechnet mit `resize=auto` die Bildhöhe aus dem Seitenverhältnis des geladenen Bildes und der verfügbaren Breite; Navigation und Außenabstände kommen dazu. Bei Fenstergrößen- oder Bildwechseln passt sich die iframe-Höhe automatisch an. Dafür sind keine festen Breiten-/Höhenwerte nötig.

Für ein Notebook im Projektordner:

```python
from widgets.stepper import display_stepper

base = "" if book else "http://127.0.0.1:8000/"
display_stepper(
    base + "widgets/2.3%20Generischer%20Bilderfolgen-Stepper/index.html",
    step=1,
    fade=0,
    theme="light",
    prefix="../../figures/4/desk_lightsources_example_",
    start=1,
    end=5,
    ext="svg",
)
```

`ui=minimal` und `resize=auto` setzt die Hilfsfunktion automatisch. Der `book`-Schalter und die Veröffentlichung der Widget-/Bilddateien funktionieren wie bisher. Wenn beim Buch-Build das Notebook aus `mynewbook` ausgeführt wird, muss auch das Python-Modul dort importierbar sein: den Ordner `widgets` vor dem Build nach `mynewbook/widgets` kopieren oder den Projektordner zum Python-Suchpfad hinzufügen. Im klassischen Notebook muss die Ausgabe vertrauenswürdig sein, damit ihr JavaScript ausgeführt wird (File → Trust Notebook). Im HTML-Buch ist der Listener Bestandteil der gespeicherten HTML-Ausgabe.

Auch im Notebook und HTML-Buch begrenzt die Hilfsfunktion die Höhe auf 90 % des Browser-Viewports (im Buch abzüglich der Artikel-Kopfzeile). So bleiben Hochkantbilder einschließlich Navigation in einer Bildschirmansicht darstellbar; kleinere Bilder behalten ihre natürliche Höhe. Im RISE-Präsentationsmodus gilt stattdessen die speziell angepasste Begrenzung anhand der Präsentationsfläche, Reveal-Skalierung und Inhalte oberhalb des Widgets auf derselben Folie. Das Bild wird bei Bedarf proportional verkleinert, die Navigation bleibt sichtbar. Beim Verlassen des Präsentationsmodus gilt wieder die normale Viewport-Begrenzung. Für die Aktualisierung in einem bereits laufenden Notebook den Kernel neu starten oder das Modul mit `importlib.reload` neu laden und anschließend die Ausgabezelle erneut ausführen.

Nur `resize=auto` an einem normalen `IPython.display.IFrame` reicht nicht: Das Eltern-Dokument benötigt ebenfalls den Listener. `display_stepper` liefert beide Seiten des Protokolls. Die Größenmeldungen funktionieren auch zwischen unterschiedlichen Ports über `postMessage`.

### Beliebige Liste über URL

Der Parameter `files` enthält eine URL-kodierte JSON-Liste von Bildadressen (oder Objekten mit `src`, `caption`, optional `alt`). Am einfachsten die Adressen im Widget einfügen, **Liste laden** klicken und anschließend den Link kopieren. `files` hat Vorrang vor `prefix`.

### Liste fest in der HTML-Datei hinterlegen

Den Inhalt von `<script id="sequence-config" type="application/json">` ersetzen:

```json
{
  "title": "Linsenprinzip",
  "frames": [
    {"src": "../../figures/imaging_pinhole_lens_0.svg", "caption": "Lochkamera", "alt": "Strahlengang durch eine Lochblende"},
    {"src": "../../figures/imaging_pinhole_lens_1.svg", "caption": "Linse", "alt": "Strahlengang durch eine Sammellinse"}
  ]
}
```

Eine einfache Liste wie `"frames": ["bild_0.svg", "bild_1.png"]` funktioniert ebenfalls. Leere `frames` aktivieren die Demo. Bei einer ungültigen Konfiguration erscheint eine Erklärung mit Demo als Rückfall. Fehlende Bilder werden mit einer Fehlermeldung angezeigt; die Navigation bleibt bedienbar.

## Jupyter Book / Web-Buch

HTML-Datei und Bilder als statische Dateien veröffentlichen. In einer Raw-HTML-Zelle bzw. einem HTML-Block:

```html
<iframe
  src="widgets/2.3%20Generischer%20Bilderfolgen-Stepper/index.html?prefix=../../figures/imaging_pinhole_lens_&amp;start=0&amp;end=5&amp;ext=svg"
  title="Interaktive Bilderfolge: Linsenprinzip"
  width="100%"
  height="850"
  style="border:0"
  loading="lazy">
</iframe>
```

Der `src`-Pfad muss zur veröffentlichten Buchseite passen. Beim Kopieren in den Build müssen auch die referenzierten Bilddateien am entsprechenden Ort vorhanden sein. Für lokale Tests im Browser genügt Doppelklick; für die Kontrolle der Web-Einbindung kann ein statischer HTTP-Server verwendet werden.

## PDF und Vorlesung

Für einen PDF-Export einen Screenshot des gewünschten, fertig geladenen Schritts verwenden und mit dem veröffentlichten Widget-Link sowie einem QR-Code versehen. Dafür gibt es den stabilen Selektor `#stage` für das Bild bzw. `.card` für Bild und Navigation. Ein Screenshot-Werkzeug kann auf `#stage[aria-busy="false"]` warten; zusätzlich sicherstellen, dass `#image-error` verborgen ist. Der PDF-Export und die QR-Erzeugung gehören zur Buch-Build-Pipeline.

Für die Vorlesung den Browser-Vollbildmodus verwenden. Bedienelemente sind touch-tauglich, Schriftgrößen mindestens 18 px; das Layout passt sich schmalen Displays an.
