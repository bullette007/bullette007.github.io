# 3.4 · Schlieren-/Winkelfilter

`index.html` direkt im Browser öffnen. Eigenständiges Offline-Widget ohne externe Bibliotheken, mit Deutsch/Englisch, Hell/Dunkel, Zustandslinks und HTML-Export. Standard: Englisch und dunkles Farbschema; Deutsch mit `?lang=de`.

## Lernweg

1. **Strahlengang:** Ein dünnes brechendes Objekt ändert die Richtung. Eine Linse bildet den Objektpunkt auf den Sensor ab und die Richtung auf eine Position in ihrer hinteren Brennebene.
2. **Brennebene:** Das kreisförmige Quellbild wird durch die Ablenkung verschoben. Die Messerschneide blockiert einen einstellbaren Anteil. Blau zeigt die ungestörte Referenz, Orange den ausgewählten Objektpunkt.
3. **Schlierenbild:** Der durchgelassene Flächenanteil ergibt die Helligkeit. Ein Klick ins Bild oder die x-/y-Regler verbinden einen Objektpunkt mit den beiden anderen Ansichten.

Messerschneide verschieben, Filternormale drehen, offene Seite umkehren, Ablenkungsstärke (auch Vorzeichen), Quellbildgröße und Brennweite ändern. Vier Presets für typische Vorführungen. Entfernen der Schneide ergibt ein weißes Bild trotz Ablenkung. **R** setzt die optischen Einstellungen zurück und erhält Sprache und Farbschema. Native Regler sind per Tastatur bedienbar.

## Modell

Kapitel 3, **Schlieren imaging / Optical setup of Schlieren imaging**. Dünnes, verlustfreies gaußförmiges Phasenobjekt, paraxiale geometrische Optik, ideale 2f–2f-Abbildung mit Vergrößerung −1. Das Bildpanel zeigt die Objektkoordinaten aufrecht zurückgeordnet, mit fester linearer Helligkeitsskala ohne Kontrastspreizung.

Mit dimensionslosen Objektkoordinaten `X = x/σ`, `Y = y/σ`, `σ = 5 mm`:

```text
α(X,Y) = −A exp((1−X²−Y²)/2) (X,Y)      [mrad]
q = f α / 1000                         [mm]
n = (cos φ, sin φ)
q∥ = n · q
```

Die größte radiale Ablenkung ist `|A|` bei `r = σ`. Positives A lenkt zur Mitte. Physikalischer Zusammenhang: `α = ∇⊥ ∫(n−n₀) dz / n₀` für kleine Ablenkungen. Quellrichtungen werden inkohärent addiert; ihre Ablenkung durch das dünne Objekt wird als richtungsunabhängig angenommen.

Das ungestörte Quellbild ist eine gleichmäßig helle Kreisscheibe mit Durchmesser `ε`, direkt in der Brennebene vorgegeben. Für `z = clamp((q∥−k)/(ε/2), −1, 1)` ist der offene Flächenanteil

```text
I/I₀ = 1/2 + (asin z + z sqrt(1−z²)) / π .
```

Umkehren liefert `1−I/I₀`; ohne Schneide gilt `I/I₀ = 1`. Die offene Halbebene ist normalerweise `n·q ≥ k`. **φ bezeichnet die Filternormale**, nicht die Schneidenrichtung: φ = 0° ergibt eine vertikale Schneide, empfindlich für αx. Bei k = 0 beträgt die Hintergrundhelligkeit 50 %. Kleinere Quellen erhöhen die Empfindlichkeit, verringern aber den nicht gesättigten Winkelbereich. Eine Änderung von f hält ε konstant; in einem realen Aufbau müsste dafür die Beleuchtungsoptik angepasst werden.

Die Strahlenskizze zeigt drei exemplarische Quellrichtungen im Schnitt entlang n; sie stehen nicht für gleiche Intensitätsgewichte. Die transversale Koordinate des gewählten Punkts außerhalb dieses Schnitts bleibt in dessen Ablenkung enthalten. Abstände entlang der optischen Achse werden normiert gezeichnet; die physikalischen 2f-/f-Abstände sind beschriftet. Keine Beugung, Absorption, Aberrationen, Vignettierung oder Kaustiken.

## Einbettung

```html
<iframe src="widgets/3.4%20Schlieren-Winkelfilter/index.html?lang=de"
        title="Schlieren und Winkelfilter" width="100%" height="1900"
        style="border:0" loading="lazy"></iframe>
```

Passende Stelle: nach dem Hinweis „Spatial filtering in the focal plane … corresponds to angular filtering“ im Abschnitt **Optical setup of Schlieren imaging**. Beobachtungsfrage: „Warum ändert das Verschieben der Schneide die Helligkeit bestimmter Ablenkungsrichtungen und schneidet nicht einfach eine Hälfte des Objektbilds ab?“

URL-Parameter: `edge` (−0,8…0,8 mm), `angle` (0…180°), `strength` (−2…2 mrad), `source` (0,04…0,8 mm), `focal` (100…300 mm), `x`, `y` (−2,5…2,5 in σ), `flip`, `removed`, `dark` (`true`/`false`), `lang` (`de`/`en`). Werte werden auf die Reglerbereiche und Schrittweiten begrenzt. URL-Parameter überschreiben den eingebetteten Exportzustand.

## Browserprüfung

Mit Playwright und Chromium:

```text
python "widgets/3.4 Schlieren-Winkelfilter/verify_browser.py"
```

Prüft die Bildwerte gegen eine unabhängige numerische Integration des kreisförmigen Quellbilds, Kontrastumkehr, Richtungsselektivität, Brennweitenabhängigkeit, Sättigung, Auswahl, Sprache, Theme, URL-Validierung, Offline-Export und mobile Überlaufgrenzen.
