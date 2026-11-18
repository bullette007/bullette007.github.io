# 3.5 · LCD/CMD-Sandkasten

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget ohne externe Bibliotheken, mit Deutsch/Englisch, Hell/Dunkel, Zustandslinks und HTML-Export. Standard: Englisch und dunkles Farbschema; Deutsch mit `?lang=de`.

## Lernweg

Links unter **Darstellungen einblenden** lassen sich Winkelverteilungen, CMD-Skalenbeiträge, LCD-Gitter und Erklärungstexte unabhängig ein-/ausblenden. Standardmäßig sind nur die Verteilungen und Skalenbeiträge sichtbar. Die Skalenbeiträge stehen direkt unter f/h, vor den LCD-Gittern. Messwerte bleiben sichtbar. Die Ansichtsauswahl bleibt bei Presets und Reset erhalten und wird in Zustandslinks sowie HTML-Export gespeichert (`showDistributions`, `showScales`, `showLCD`, `showDetails`, jeweils `true`/`false`).

1. Zwei **9×9-Winkelverteilungen** verändern: Ziel f oder h wählen, Peak verschieben, Breite, Gesamtintensität und reproduzierbares Rauschen einstellen. Fünf Presets starten typische Experimente.
2. Position und **Fensterradius b** auswählen: Der orange Rahmen zeigt die summierten Zellen. Daneben erscheinen beide LCDs, ihre vorzeichenbehaftete Differenz und die Rechnung für den ausgewählten Ort.
3. **L₂, L₂² und CMD** in der oberen, beim Scrollen sichtbaren Messwertleiste vergleichen. Die Balken zeigen den CMD-Beitrag jedes Radius; anklicken wählt die entsprechende LCD-Ansicht. Die Teilsumme bis zum ausgewählten Radius wird zusätzlich angezeigt. Die Detailbox „Ein Fenster, zwei lokale Summen“ steht unter den LCD-Gittern.
4. Optional auf Summe 1 normieren: Damit verschwindet die Empfindlichkeit gegenüber reinen Intensitätsskalierungen.
5. Optional direkt in f und h malen. Jeder Klick **addiert den Pinselschritt** zum rohen Zellwert, maximal bis 2. Pinselschritt 0 löscht eine Zelle. Modellregler oder „Gaußprofil neu erzeugen“ ersetzen die manuelle Bearbeitung der gewählten Verteilung.

Klick in ein Gitter wählt einen Ort, Pfeiltasten bewegen die Auswahl. Im Malmodus addieren Klick und Leertaste in f/h den Pinselschritt. Auch bei aktiver Normierung wird der rohe Zellwert erhöht; angezeigt wird anschließend die normierte Verteilung. Koordinaten und b sind auch über native Regler erreichbar. **R** setzt zurück und erhält Sprache und Farbschema.

## Diskretes Modell

Entspricht dem Abschnitt **Generalized Cramér-von Mises Distance** in Kapitel 3:

```text
Ω = {0,…,8}²
F(x,b) = Σ_{t∈Ω, |tₓ−x|≤b, |tᵧ−y|≤b} f(t)
C_b = Σ_{x∈Ω} (F(x,b)−H(x,b))²
CMD = Σ_{b=0}^8 C_b
L₂² = Σ_{x∈Ω} (f(x)−h(x))² = C_0
C_8 = 81 (Σf−Σh)²
```

**b ist ein Radius**, die volle Seitenlänge beträgt `2b+1`. Fenster werden am Gitterrand abgeschnitten; keine Mittelung, keine periodischen Ränder, keine Wurzel des CMD. Orts- und Skalenbeiträge haben Gewicht 1. LCD-Berechnung mittels Summed-Area Tables. Die Ergebnisse enthalten damit bereits den Übergang zu fastCMD.

Die synthetischen Gaußprofile werden auf dem endlichen Gitter auf Summe 1 gebracht; `σ=0` ergibt einen Einzelpeak. Rauschen addiert unabhängige, deterministische Gleichverteilungen in `[−η/81,+η/81]`, setzt negative Werte auf 0 und normiert erneut, bevor die Gesamtintensität multipliziert wird. Seeds: f=17, h=83. Kein physikalisches Sensormodell. Verbreitern und Rauschen erhalten die vorgegebene Gesamtintensität; beim Malen ergibt sie sich aus den Zellwerten.

Optionale Normierung erfolgt vor sämtlichen Anzeigen und Berechnungen. Nullverteilungen bleiben 0 und werden ausdrücklich gekennzeichnet. f/h teilen eine **Wurzel-Farbskala**: Der Farbmischfaktor ist `sqrt(Wert / gemeinsames Maximum)`. Dadurch bleiben breite Gaußprofile trotz erhaltener Gesamtintensität sichtbar. Gleiche Intensitäten erhalten in beiden Gittern dieselbe Farbe; die Zahlenwerte und Abstandsrechnungen bleiben unverändert. F/H teilen eine zweite, lineare Skala, deren Maximum die größere Gesamtsumme ist und damit beim Wechsel von b konstant bleibt. Die Differenzskala ist linear, symmetrisch und separat beschriftet. Zahlen lassen sich ausblenden; Zellwerte bleiben als Tooltips zugänglich.

CMD und L₂ haben unterschiedliche Skalen. Ihre absoluten Zahlengrößen sind kein Qualitätsvergleich. Eine leichte lokale Veränderung ist nicht für jedes Signal und jedes Rauschen garantiert robuster; Fenster können Differenzen räumlich ausmitteln. Randbeschnitt beeinflusst die Ortsabhängigkeit.

## Einbettung

```python
from IPython.display import IFrame, display
display(IFrame(
    base + "widgets/3.5%20LCD-CMD-Sandkasten/index.html?lang=de",
    width="100%", height=1800,
))
```

Passende Stelle: nach der LCD-Beispielanimation und vor der Komplexität der naiven CMD-Berechnung. Beobachtungsfrage: „Warum bleibt der L₂-Abstand zweier gleich heller Einzelpeaks für verschiedene nichtverschwindende Verschiebungen gleich, während sich der CMD ändert?“

Auf schmalen Bildschirmen stehen die Panels untereinander; der iframe bleibt scrollbar. Kapitelquelle und Notebook enthalten die Einbettung.

URL-Parameter: `fx`, `fy`, `hx`, `hy` (−4…4, relative Peakposition), `fsigma`, `hsigma` (0…2), `fmass`, `hmass` (0…2), `fnoise`, `hnoise` (0…1), `x`, `y`, `b` (0…8), `brush` (0…1), `target` (`f`/`h`), `normalize`, `numbers`, `paint`, `dark` (`true`/`false`), `lang` (`de`/`en`). `customF` und `customH` können JSON-Arrays mit je 81 nichtnegativen Zellwerten bis 2 enthalten. Ungültige Arrays werden verworfen. URL-Parameter überschreiben den eingebetteten Exportzustand; Reglerwerte werden begrenzt und gerundet.

## Browserprüfung

Mit Playwright und Chromium:

```text
python "widgets/3.5 LCD-CMD-Sandkasten/verify_browser.py"
```

Prüft die LCDs und Abstände gegen unabhängige direkte Fenstersummen, Randbeschnitt, Einzelpeaks, Intensität und Normierung, Rauschen, Nullverteilungen, Malen, Tastaturauswahl, Sprache, Theme, URL-Rundlauf, Offline-Export und mobile Überlaufgrenzen.
