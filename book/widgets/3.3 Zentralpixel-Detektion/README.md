# 3.3 · Detektion zentraler Displaypixel

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget im Stil der bisherigen Widgets: Deutsch/Englisch, Hell/Dunkel, Zustandslinks und HTML-Export mit aktuellem Zustand. Standard: Englisch, dunkles Farbschema. Deutsch über `index.html?lang=de`.

Visualisiert **Determination of central pixels / acquireCentralPixels** aus Kapitel 3 (`03_LightFieldMethods.md`), als horizontalen 1D-Schnitt.

## Szene und Algorithmus

- Drei Mikrolinsen mit jeweils fünf Displaypixeln: **15 global nummerierte Displaypixel** `u = 1 … 15`.
- Vier Kamerapixel je Mikrolinse: **12 Kamerapixel** `m = 1 … 12`.
- Zentrale Displaypixel: **3, 8, 13**. Ihr Licht wird achsparallel kollimiert. Die vier Kamerapixel einer Mikrolinse lösen deren Apertur auf und sehen denselben zentralen Quellpixel, erhalten also identische Codes.
- Vier **globale Binärmuster**, vom niederwertigsten zum höchstwertigen Bit: Gewichte **1, 2, 4, 8**. Die Bitmuster werden nicht pro Mikrolinse neu gestartet.
- Dunkelreferenz `g₀ = 0,1`, Nutzsignal `0` oder `1`, Schwellwert `t = 0,5`. Entscheidung nach Dunkelbildabzug: `g > t → 1`, sonst `0`.
- Ergebnis: `0011₂ = 3`, `1000₂ = 8`, `1101₂ = 13` für die drei Kamerapixelgruppen.

Die telezentrische Kamera ist als idealer Richtungsfilter mit schematischer aufrechter Abbildung dargestellt. Das Widget zeichnet vier räumliche Abtastungen des zentralen Bündels und optional eine exemplarische verworfene Richtung pro Mikrolinse. Es ist kein maßstäblicher Objektiventwurf. Die vertikale `v`-Serie folgt in der vollständigen Kalibrierung analog; hier wird nur `g_hor` rekonstruiert.

## Bedienung

1. **Schritt 0–1:** Display dunkel, Dunkelreferenz aufnehmen und Speicher initialisieren.
2. **Schritt 2–13:** Pro Bit drei getrennte Schritte: **Pattern anzeigen → Kamera aufnehmen → Bit speichern**.
3. **Schritt 14:** Gespeicherte Codes dekodieren und zentrale Displaypixel markieren.

Die Navigationsbuttons springen zum Beginn einer Bitmessung. Vor/Zurück und der Schieberegler erlauben alle Einzelschritte. **Zurückgehen stellt den Speicher auf diesen Schritt zurück**; zukünftige Bits werden wieder unbekannt (`·`). Nullbits werden explizit als `0` angezeigt. Die Zahlen in den Kamerafeldern zeigen den aktuellen Lichtzustand. Der kompakte Bit-Speicher sitzt direkt darüber, spaltengenau an den zwölf Kamerapixeln ausgerichtet, und zeigt nur tatsächlich gespeicherte Bits sowie abschließend die dekodierte Koordinate.

Unter jeder Displaykoordinate steht ihr vierstelliger Binärcode in der Reihenfolge **b3 b2 b1 b0**. Die aktuell projizierte Bitposition wird violett hervorgehoben, sodass die Verbindung zwischen Koordinate, Bitmuster und Displayzustand direkt sichtbar ist.

Kamerapixel per Regler, Klick auf eine Speicherspalte oder Klick im Sensor auswählen; die Speicherspalten sind auch per Tab und Enter/Leertaste bedienbar. Der Messwert und der entstehende Code werden darunter erläutert. Zentrale Pixel lassen sich vorzeitig markieren. Wiedergabe: 1,5 Sekunden je Schritt, stoppt beim Ergebnis. Am Ende startet erneutes Abspielen von vorn.

- **← / →:** Einzelschritt; **Leertaste:** Abspielen/Pause; **R:** Reset. Fokussierte Bedienelemente behalten ihre Tastaturbedienung.
- Auf schmalen Bildschirmen ist die gemeinsame Strahlenskizze mit Speicher und Binärcodes horizontal scrollbar, damit alle Pixel lesbar bleiben.
- **Zustandslink kopieren** mit auswählbarem Link als Clipboard-Fallback.
- **Widget mit Zustand speichern** exportiert eine eigenständige HTML-Datei; die Wiedergabe startet beim Öffnen pausiert.

## Einbettung

```html
<iframe
  src="widgets/3.3%20Zentralpixel-Detektion/index.html?lang=de"
  title="Detektion zentraler Displaypixel"
  width="100%"
  height="2100"
  style="border:0"
  loading="lazy">
</iframe>
```

URL-Parameter: `step` (0–14), `pixel` (1–12), `rejected` (`true`/`false`), `reveal` (`true`/`false`), `lang` (`de`/`en`), `dark` (`true`/`false`). URL-Parameter überschreiben den eingebetteten Exportzustand.

Beispiele:

```text
index.html?lang=de&step=2
index.html?lang=de&step=6&pixel=6
index.html?lang=de&step=14&dark=false
```

## Browserprüfung

Mit installiertem Playwright und Chromium:

```text
python "widgets/3.3 Zentralpixel-Detektion/verify_browser.py"
```

Prüft die globalen Muster, die Kameragruppen, den getrennten Aufnahme-/Speichervorgang, alle vier Bitgewichte, Dekodierung, Rückwärtsnavigation, Wiedergabe, Auswahl, DE/EN, Theme, URL-Validierung, Offline-Export und mobile Überlaufgrenzen. Screenshots werden im temporären OpenCode-Verzeichnis gespeichert.
