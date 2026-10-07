# Computational Imaging — Ideen für interaktive HTML5-Visualisierungen und Aktivierungsformate

**Grundlage:** Vorlesungsunterlagen `computational-imaging.de` (PDF-Export der Kapitel 1–12, Stand August 2026)
**Umfang:** Ideensammlung, keine Umsetzung
**Adressat:** Dr.-Ing. Johannes Meyer, KIT / Fraunhofer IOSB

---

## Inhalt

1. [Ausgangslage und Befund](#1-ausgangslage-und-befund)
2. [Technische Leitplanken](#2-technische-leitplanken-für-alle-widgets)
3. [Widget-Ideen nach Kapiteln](#3-widget-ideen-nach-kapiteln)
4. [Kapitelübergreifende Widgets](#4-kapitelübergreifende-widgets)
5. [Priorisierung](#5-priorisierung--was-zuerst)
6. [Aktivierung der Studierenden während der Vorlesung](#6-aktivierung-der-studierenden-während-der-vorlesung)
7. [Konkrete ConcepTest-Fragen zum Sofort-Einsetzen](#7-konkrete-conceptest-fragen-zum-sofort-einsetzen)
8. [Verzahnung mit Übung und mündlicher Prüfung](#8-verzahnung-mit-übung-und-mündlicher-prüfung)
9. [Nebenbefunde aus der Materialdurchsicht](#9-nebenbefunde-aus-der-materialdurchsicht)

---

## 1. Ausgangslage und Befund

### Was im Material bereits interaktiv angelegt ist

Die Notebooks enthalten im Wesentlichen zwei Sorten von `ipywidgets`-Interaktion:

| Typ | Mechanik | Anzahl (grob) | Zustand im PDF/Web |
|---|---|---|---|
| **Bilderfolgen-Stepper** | `interact(lambda i: showFig('figures/…_',i,'.svg'), IntSlider)` — durchklickbare Aufbau-Animationen von Zeichnungen | ~25 Stellen | tot, es wird nur der Endzustand (`max_i if book`) gezeigt |
| **Live-Rechnung** | matplotlib-Plot wird bei Reglerbewegung neu berechnet (Dirac-ε, Fourier-Reihe, Defokus-Faltung, MTF, Sigmoid-Neuronen, Soft-Thresholding, Fourier-Filterung) | ~10 Stellen | tot, nur ein Standbild |
| **JS-Injection-Stepper** (Kap. 12) | `Javascript(...)` mit `localhost:8888`-URL | 5 Stellen | **doppelt tot** — „JavaScript output is disabled in JupyterLab" steht sogar im PDF |

Die Kapitel-12-Variante ist besonders fragil: sie hängt an einem laufenden Notebook-Server auf `localhost:8888` inklusive XSRF-Token. Das ist genau die Stelle, an der ein Wechsel auf eigenständiges HTML den größten Stabilitätsgewinn bringt.

### Der eigentliche Hebel

Auffällig ist, dass der inhaltliche Kern der Vorlesung fast durchgehend **aus veränderlichen Größen besteht**, die im Skript als feste Zahl oder festes Bild erscheinen:

- $b'/b = \alpha$ (Refokussierungstiefe), $\lambda, \rho$ (HQS/ADMM), $\sigma$ (Rauschpegel), $p, K$ (Bernoulli-Probing), $N_F$ (Fresnel-Zahl), $\gamma$ (Sigmoid-Binarisierung), $b$ (LCD-Kernel), $r$ (Zerstreuungskreis) …
- und an vielen Stellen ganze **Entwurfsobjekte**: der 52-Bit-Shutter-Code, die Blendenmaske, die Deflektometrie-Musterfolge, die PSF der Amplitudenmaske.

Ein Skript kann nur einen Punkt aus diesem Parameterraum zeigen. Eine Vorlesung, in der Sie den Regler ziehen, zeigt die *Ableitung* — also genau das, was Studierende in der mündlichen Prüfung brauchen („Was passiert, wenn …?"). Deshalb sind die Widget-Vorschläge unten fast alle so gebaut, dass sie **eine Behauptung des Skripts überprüfbar machen**, statt sie nur zu illustrieren.

### Drei Sorten von Vorschlägen

Zur besseren Einordnung sind alle Ideen unten markiert:

- 🔁 **Ersatz** — bildet eine bestehende (tote) Interaktion 1:1 in HTML nach, geringer Aufwand
- ➕ **Erweiterung** — nimmt eine bestehende Stelle und macht sie deutlich mächtiger
- ✨ **Neu** — eine Visualisierung, die es im Material bisher gar nicht gibt und eine echte Verständnislücke schließt

Aufwandsschätzung: **S** ≈ halber Tag, **M** ≈ 1–3 Tage, **L** ≈ eine Woche+ (bzw. Kandidat für eine studentische Arbeit / HiWi-Projekt).

---

## 2. Technische Leitplanken (für alle Widgets)

Bevor die Ideen kommen — ein paar Entscheidungen, die sich durch alle ziehen und die man einmal am Anfang treffen sollte:

**Eigenständige HTML-Dateien statt Notebook-Zellen.**
Eine Datei = ein Widget, keine externen Abhängigkeiten außer ggf. einer kleinen FFT-Bibliothek. Vorteile: läuft im Hörsaal ohne Netz, läuft auf dem Tablet der Studierenden, läuft in fünf Jahren noch, und die Jupyter-Book-Toolchain muss nichts davon wissen.

**Einbindung dreifach:**
- **Web-Buch:** `<iframe>` in einer Raw-HTML-Zelle. Damit ist das Widget im Online-Skript *live* — genau der Punkt, der Ihnen aktuell fehlt.
- **PDF:** automatisch generiertes Standbild (Puppeteer-Screenshot des Default-Zustands) + QR-Code + Kurzlink. Sie nutzen mit `s.fhg.de/optics-simulator`, `s.fhg.de/convolution`, `s.fhg.de/fourier`, `s.fhg.de/wave-game` bereits genau dieses Muster für externe Applets — das eigene Material sollte konsistent dazu adressierbar sein.
- **Vorlesung:** Vollbild im Browser, zweiter Bildschirm.

**Zustand in der URL.**
Jeder Reglerstand als URL-Parameter (`?psf=pillbox&r=7&sigma=0.03&method=wiener`). Das kauft drei Dinge auf einmal:
1. Sie können im Skript auf eine *bestimmte* Konfiguration verlinken („vergleiche mit dieser Einstellung").
2. Sie können sich für die Vorlesung Lesezeichen für Ihre Demo-Punkte anlegen und springen im Hörsaal nicht mit dem Regler herum.
3. Studierende können ihren Zustand teilen — Voraussetzung für die Wettbewerbsformate weiter unten.

**Ein gemeinsames Farb- und Notationsschema über alle Widgets.**
Vorschlag: Szene/Latentbild $s$ blau, Messung $g$ orange, Rekonstruktion $\hat{s}$ grün, PSF/Kernel $h$ violett, Rauschen $n$ grau. Formelsymbole exakt wie im Skript. Der Wiedererkennungswert über zwölf Kapitel hinweg ist erheblich — die Studierenden sollen bei Kapitel 11 sofort sehen, dass da wieder dasselbe $g = Hs + n$ steht.

**Rechenleistung realistisch einplanen.**
- 2D-FFT bei 256×256 in reinem JS: unproblematisch, ~60 fps.
- 512×512 mit iterativem ADMM/RL: WebGL oder WebGPU nötig, oder vorberechnete Frames.
- **Pragmatischer Zwischenweg für teure Verfahren:** Ergebnisse offline in Python berechnen (Sie haben den Code ja bereits), als Bildkacheln exportieren, im Widget nur durchblenden. Sieht für die Studierenden identisch aus und kostet einen Bruchteil der Entwicklungszeit. Das ist besonders bei Kapitel 6 und 7 der richtige Weg für den ersten Wurf.

**Beamer-Tauglichkeit von Anfang an:** Schriftgrößen ≥ 18 px, Reglerknöpfe groß genug für Touchpad-Bedienung unter Stress, Tastatursteuerung (Pfeiltasten = Regler, `R` = Reset), hell/dunkel umschaltbar. Ein „Reset"-Knopf ist im Live-Betrieb Gold wert.

**Deterministische Zufallszahlen** (fester Seed, sichtbar, änderbar). Sonst zeigen Sie in der Vorlesung ein anderes Rauschbild als im Skript und müssen es erklären.

---

## 3. Widget-Ideen nach Kapiteln

### Kapitel 2 — Fundamental Basics

Dieses Kapitel ist der Werkzeugkasten für alles Weitere. Widgets, die hier gebaut werden, zahlen sich in fünf späteren Kapiteln erneut aus — entsprechend hoch sollte die Investition sein.

---

**2.1 · PSF ↔ MTF-Doppelansicht** ➕ · **M** · ⭐ *Top-Kandidat*

Drei gekoppelte Panels: links die PSF $h(\mathbf{x})$, rechts die MTF $|\mathcal{F}\{h\}|$, unten ein Testbild gefaltet mit $h$.

PSF wählbar aus: Dirac (in-focus), Pillbox mit Radius-Regler (Defokus), Gauß, Rect (Bewegung), 1D-codiertes Rect, freie Blendenform mit der Maus zeichenbar.

Der entscheidende Zusatz: **Nullstellen der MTF werden rot markiert**, mit Zähler („7 Nullstellen im dargestellten Bereich"). Beim Umschalten von Pillbox auf codiertes Rect verschwinden sie sichtbar.

*Warum das so viel bringt:* Diese eine Ansicht trägt die Argumentation in **vier** späteren Kapiteln — Kap. 6 (warum der Inversfilter explodiert), Kap. 7 (Designziel „flaches Magnitudenspektrum" für lensless-Modulatoren), Kap. 8 (warum das rect-Kernel schlecht und der Raskar-Code gut ist), Kap. 12 (die Frage „Wie bekommen wir systemtheoretischen Einblick, was diese PSFs können?" wird im Skript gestellt und mit zwei MTF-Standbildern beantwortet). Ein Widget, vier Einsatzorte.

---

**2.2 · Dünne-Linse-Sandkasten mit Bildsimulation** ➕ · **M**

Strahlendiagramm mit Reglern für $f$, $g$, $b$, Blendendurchmesser $d$, Sensorposition. Live: Hauptstrahl, Parallelstrahlen, Zerstreuungskreis mit angezeigtem Durchmesser, Vergrößerung $V$, und — der Mehrwert gegenüber bestehenden Lösungen: **das resultierende Bild** eines Testmotivs daneben.

Zusatz-Toggle „Zerstreuungskreis vs. Pixelgröße": die Schärfentiefe wird als Bereich in der Objektebene eingeblendet, sobald der Zerstreuungskreis unter die Pixelgröße fällt. Damit wird die Definition aus dem Skript zur sichtbaren Grenze statt zu einem Satz.

*Anschluss:* Der Regler „Sensorposition" ist bereits das gedankliche Vorspiel zur digitalen Refokussierung in Kap. 3 („virtuelles Verschieben des Sensors"). Wenn Sie in Kap. 3 dasselbe Widget nochmal aufrufen und sagen „das konnten wir nur *vorher* — jetzt können wir es *nachher*", sitzt die Pointe.

---

**2.3 · Generischer Bilderfolgen-Stepper** 🔁 · **S** · ⭐ *bester Aufwand-Nutzen-Schnitt im ganzen Dokument*

Ein einziges HTML-Widget, das eine Liste von SVG/PNG-Dateien bekommt und mit Pfeiltasten/Klick durchblättert — mit Fortschrittsanzeige und optionalem Crossfade.

Damit sind **alle ~25 `showFig`-Stellen** auf einen Schlag im Web-Buch wiederbelebt: `imaging_pinhole_lens_`, `dftSpectrum_`, `lfCameraPrinciple_`, `lfExample_`, `SchlierenPrinciple_`, `lfg_concept_`, `deflectionMapAcquisition_`, `exampleDeflMap_`, `LCDExample_`, `LightfieldLaserScannerWithObject_`, `invLF_acquisition_`, `invLF_emission_`, `desk_lightsources_example_`, `symmetric_transport_matrix_`, `pooling_layer_`, `convolutional_layer_`, `transposed_convolutional_layer_`, `inv_filter_res_`, `wiener_filter_res_`, `dw_ex_`, `diffuser_cam_shift_invaraince_`, `diffuser_cam_grad_recon_`, `diffuser_cam_ADMM_recon_`, `motion_deblur_results_`, `transient_imaging_`, `nlos_task_`, `end2end_general_`, `selaci_*`, `deflecto_*`.

Ein halber Tag Arbeit, und das Online-Skript fühlt sich schlagartig lebendig an. **Wenn Sie nur eine Sache aus diesem Dokument umsetzen, dann diese.**

---

**2.4 · Faltungs-Scrubber** ➕ · **S**

$f(x)$ und $h(x)$ aus einer Palette wählbar oder mit der Maus zeichenbar; Regler für die Verschiebung $\alpha$; dargestellt: gespiegeltes und verschobenes $h$, die Überlappungsfläche schraffiert, und das sich Punkt für Punkt aufbauende Ergebnis.

Sie verlinken aktuell auf `lpsa.swarthmore.edu`. Ein eigenes Widget lohnt sich trotzdem, weil (a) die Notation Ihre ist, (b) Sie einen 2D-Modus anhängen können — Kernel über Bild wandernd, mit Kachelvergrößerung — und (c) Sie den Übergang zum Impulsantwort-Argument ($f * \delta$) direkt als Preset anbieten können.

---

**2.5 · 2D-FFT-Malkasten** ➕ · **M**

Bild links, Log-Magnitudenspektrum rechts. Mit der Maus im Spektrum malen (Pinsel/Radierer) → Bild aktualisiert sich live. Und umgekehrt.

Presets: Tiefpass, Hochpass, Bandpass, Richtungsfilter, **Pillbox** (= genau Ihr `fourier_filtering(r)`-Beispiel), Bewegungs-Rect. Ein Toggle „fftshift an/aus" macht den zweiten Punkt Ihrer DFT-Notiz („die Mitte des Fensters entspricht den höchsten Frequenzen") in einem Klick sichtbar — das ist erfahrungsgemäß eine der zuverlässigsten Fehlerquellen bei Studierenden.

---

**2.6 · Telezentrie-Umschalter** ✨ · **S**

Objekt in der Tiefe verschiebbar, Blende in der Brennebene ein-/ausschaltbar. Ohne telezentrische Blende: Bildgröße ändert sich. Mit: bleibt konstant. Daneben ein Marker, der die Bildhöhe misst.

Klein, aber das telezentrische Prinzip taucht danach in **Kap. 3** viermal auf (Kalibrierung des Lichtfeld-Displays, Schlierendeflektometer, 4f-Kamera, inverse Lichtfeldbeleuchtung). Wenn es hier sitzt, sparen Sie später vier Erklärungen.

---

**2.7 · Blenden- und Bokeh-Sandkasten** ✨ · **S**

Blendenform wählbar (Kreis, Sechseck, Herz, codierte Binärmaske), Punktlichtquellen in verschiedenen Tiefen. Auf dem Sensor erscheint die Blendenform als Bokeh. Regler „Abblenden" verkleinert Bokeh und vergrößert Schärfentiefe.

Direkter Vorlauf zu Coded Aperture Imaging (Kap. 1 Motivation, Kap. 2 Ausblick) — und ein ausgesprochen dankbares Bild, weil jede*r es aus der eigenen Fotografie kennt.

---

**2.8 · Dirac-Approximation & Fourier-Reihe** 🔁 · **S** (je)

Die bestehenden `plotDiracEps(eps)`, `reconstrTrigFourier(terms)` und `reconstrExpFourier(terms)` direkt als HTML. Bei der Fourier-Reihe zwei Erweiterungen, die wenig kosten und viel bringen:
- **Zielsignal mit der Maus zeichnen** statt nur Sägezahn
- **Koeffizientenspektrum** $a_f, b_f$ bzw. $|c_f|$ als Balkendiagramm daneben, einzelne Harmonische zuschaltbar

Sie verlinken auf `falstad.com/fourier` und das PhET-Wave-Game — beide sehr gut. Ein eigenes lohnt vor allem wegen des gezeichneten Signals („zeichne deinen Namen und schau, wie viele Terme du brauchst").

---

### Kapitel 3 — Light Field Methods

Das längste und bilderreichste Kapitel (drei bis vier Vorlesungstermine). Hier liegt das größte ungenutzte Potenzial, weil Lichtfelder von Natur aus vierdimensional sind und ein Papierausdruck bestenfalls zwei Dimensionen zeigt.

---

**3.1 · Lichtfeld-Explorer (SAI-Navigator + Refokussierung)** ✨ · **L** · ⭐ *Top-Kandidat*

Ein echtes Lichtfeld aus dem Stanford-Archiv als Bildstapel (z. B. 9×9 Sub-Aperture-Bilder à 256×256 in einem Sprite-Sheet, ~2–4 MB als JPEG).

Bedienung:
- **Maus über dem Bild** → Blickwinkel $(u,v)$ ändert sich, Parallaxe wird sofort körperlich spürbar
- **Regler $\alpha$** → Shift-and-Add-Refokussierung, live im Canvas gerechnet (das ist billig: Verschieben und Mitteln von 81 Bildern)
- **Modus-Umschalter:** einzelnes SAI (= kleine Blende, viel Schärfentiefe, verrauscht) ↔ refokussiert (= volle Blende) ↔ EDoF-Fusion
- **Anzeige des Rauschpegels**, damit der im Skript genannte Nachteil des SAI-Ansatzes („weniger Licht, mehr Rauschen") nicht nur behauptet wird

Sie haben mit `lfExample_` (8 Stufen) und `lfSAIs/out_09_` (17 Stufen) bereits Bildmaterial für die statische Variante. Der Sprung zum echten Explorer ist die Investition wert — das ist das Widget, an das sich die Studierenden im nächsten Semester noch erinnern.

---

**3.2 · Das (x,u)-Lichtfeld-Diagramm, gekoppelt an den Strahlengang** ✨ · **M** · ⭐ *Top-Kandidat*

**Links:** Strahlendiagramm mit Szenenpunkt, Linse, Sensor. **Rechts:** das zugehörige $(x,u)$-Lichtfeld-Diagramm.

- Szenenpunkt mit der Maus in der Tiefe verschieben → die Gerade im Lichtfeld-Diagramm kippt. Vor der Fokusebene: positive Steigung. Dahinter: negative. Genau die drei Bildpaare aus dem Skript, aber kontinuierlich und selbst gesteuert.
- **Integrationsrichtung** als drehbare Linienschar einblendbar → „so entsteht $g_b$", „so entsteht $g_{b'}$".
- $\alpha$-Regler koppelt Neigung und Refokussierung.

*Warum das wichtig ist:* Die Kernformel des Kapitels

$$g_{b'}(x',y') = \iint L_b\!\left(u\left(1-\tfrac{1}{\alpha}\right)+\tfrac{x'}{\alpha},\ v\left(1-\tfrac{1}{\alpha}\right)+\tfrac{y'}{\alpha},\ u,v\right) du\, dv$$

ist für die meisten Studierenden eine Wand aus Symbolen. In der $(x,u)$-Ebene ist sie eine gekippte Gerade, über die integriert wird. Das ist der Moment, an dem „shift and add" von einem Rezept zu etwas Verstandenem wird. **Diese Visualisierung fehlt im Skript vollständig** und ist meiner Einschätzung nach die größte einzelne Verständnislücke des Kapitels.

---

**3.3 · Mikrolinsen-Designer** ➕ · **S**

Regler $d_{ML}$, $b_{ML}$, $d_L$, $b_L$. Gezeichnet: Strahlenbündel durch zwei benachbarte Mikrolinsen. **Crosstalk wird rot markiert**, sobald $d_{ML}/b_{ML} \neq d_L/b_L$. Gleichzeitige Anzeige der resultierenden Orts- und Winkelauflösung.

Zusatzregler „Sensorgröße fix" macht den Orts-Winkel-Trade-off zum Nullsummenspiel — man sieht die eine Zahl steigen, während die andere fällt.

---

**3.4 · Deflection-Map-Explorer** ➕ · **M**

Links das Prüfobjekt (Zylinderlinse, Scheinwerferabdeckung, Glasgob — Sie haben die Messdaten). Mauszeiger über eine Position $\mathbf{m}$ → **rechts erscheint die 2D-Winkelverteilung $a(\mathbf{m}, \cdot)$ an dieser Stelle.**

Der zentrale Aha-Moment: über defektfreiem Material wandert der Peak sanft; über einem Streudefekt zerfällt er schlagartig in eine breite Verteilung. Genau das, was Ihre Gradientenformel misst — und was aus statischen Bildern kaum hervorgeht.

Ergänzung: Umschalter für das Distanzmaß $d\{\cdot,\cdot\}$ (naiver L2 vs. CMD) mit dem resultierenden Inspektionsbild und **live berechnetem CNR**. Damit wird Ihre Tabelle (CNR 1,83 → 23,38) zur eigenen Beobachtung statt zu einer Zahl, die man glauben muss.

---

**3.5 · LCD- und CMD-Rechner zum Selbstbauen** ✨ · **M**

Zwei kleine Winkelverteilungen auf einem 8×8-Gitter, per Klick editierbar (Helligkeit malen). Regler für die Kernelgröße $b$. Angezeigt: beide LCDs $F(\mathbf{x},b)$, $H(\mathbf{x},b)$, ihre Differenz, und der über $b$ aufsummierte CMD-Wert.

**Die eigentliche Aufgabe für die Studierenden:** Ihre fünf Anforderungen an $d\{\cdot,\cdot\}$ stehen im Skript als Behauptungsliste („Criterion met: …"). Mit dem Widget lässt sich jede einzeln nachbauen —
1. verschobener Peak → CMD hoch?
2. verbreiterter Peak → CMD hoch?
3. skalierte Intensität → CMD ≠ 0?
4. leicht verrauschter Peak → CMD niedrig?

Vier Klicks statt vier Absätze zum Glauben. Als Kleingruppenaufgabe in der Vorlesung: „Gruppe A prüft Kriterium 1, Gruppe B Kriterium 2 …" (siehe Abschnitt 6, Jigsaw).

---

**3.6 · Summed-Area-Table-Spiel** ✨ · **S**

8×8-Zahlengitter, daneben die SAT. Mit der Maus ein Rechteck aufziehen → die vier relevanten SAT-Einträge leuchten auf, die Formel $s(m_t,n_t)-s(m_f{-}1,n_t)-s(m_t,n_f{-}1)+s(m_f{-}1,n_f{-}1)$ wird mit den konkreten Zahlen eingeblendet, das Ergebnis wird gegen die tatsächliche Summe geprüft.

Ihre Skript-Grafik erklärt das „braun und blau abziehen, grün einmal wieder addieren"-Argument mit Farbflächen — genau das, was von Selbstausprobieren enorm profitiert. Und die Komplexitätsaussage $O(1)$ wird von einer Behauptung zu etwas Handgreiflichem.

---

**3.7 · Schlieren-Baukasten** ➕ · **M**

Kollimierte Beleuchtung, Phasenobjekt wählbar (Kerzenflamme, Feuerzeuggas — Sie haben das Foto —, Glasfehler), Schlierenblende in der Brennebene mit der Maus verschiebbar: Messerschneide (Position und Winkel!), Lochblende, Farbrad.

Der Kernsatz des Abschnitts — *Ortsfilterung in der Brennebene = Winkelfilterung der Strahlen* — wird durch das Verschieben der Messerschneide direkt erfahrbar: Kante nach links → eine Ablenkungsrichtung wird hell, die andere dunkel. Beim Farbrad: Ablenkungsrichtung wird zu Farbe.

Nebenbei ist das die natürliche Stelle, um `figures/3/SchlierenPrinciple_` (5 Stufen) und die Bedingung $\delta_\alpha > \frac{1}{2}\varepsilon$ als einblendbare Grenze unterzubringen.

---

**3.8 · Inverse Lichtfeldbeleuchtung, zweistufig** ➕ · **M**

Phase 1: defektfreies Objekt, kollimierte Beleuchtung, Lichtfeld $L'$ wird „aufgenommen" (visualisiert als Pfeilfeld).
Phase 2: $\tilde{L}'$ wird emittiert, Prüfling wählbar (defektfrei / dejustierte Linse / Streudefekt / Kratzer). Kamerabild rechts.

Der Effekt — defektfreie Bereiche leuchten, Defekte bleiben schwarz, und das mit **einer einzigen Aufnahme** — ist visuell so stark, dass er einen eigenen Regler verdient. Aktuell sind das `invLF_acquisition_` (2 Stufen) und `invLF_emission_` (4 Stufen) plus Ergebnisbilder.

---

### Kapitel 4 — Light Transport Analysis

Das konzeptuell abstrakteste Kapitel („optische lineare Algebra"). Gleichzeitig das, in dem eine gute Visualisierung am meisten rettet, weil $\mathbf{T}$ prinzipiell nicht darstellbar ist — und genau das ist ja der Punkt.

---

**4.1 · Lichttransportmatrix-Sandkasten** ✨ · **L** · ⭐ *Top-Kandidat*

Eine kleine, vorberechnete Szene (z. B. Ihre Schreibtischszene, oder eine Ecke mit zwei Wänden, einem Spiegel und einem Glas trüber Flüssigkeit) mit einer echten, offline gerenderten Transportmatrix in reduzierter Auflösung, etwa $64\times64$ Projektorpixel auf $64\times64$ Kamerapixel.

Bedienung:
- **Beleuchtungsmuster $\mathbf{p}$ mit der Maus malen** → $\mathbf{i} = \mathbf{T}\mathbf{p}$ erscheint sofort. Das ist synthetisches Relighting, und es fühlt sich an wie Zaubern.
- **Umgekehrte Richtung:** Kamerapixel $m$ anklicken → die Zeile $\mathbf{T}[m,\cdot]$ wird als Bild über dem Projektorraum eingeblendet („welche Projektorpixel tragen zu diesem Kamerapixel bei?"). Bei einem Pixel im Schatten sieht man dann sehr deutlich, dass nur indirekte Pfade beitragen.
- **Helmholtz-Reziprozität** als Toggle: $\mathbf{T}$ und $\mathbf{T}^\intercal$ übereinanderlegen.

Zusätzlich ein Zähler, der Ihre Rechnung aus dem Skript live mitführt: „bei dieser Auflösung: 4096 Messungen, 2,3 Minuten bei 30 Hz — bei 1 MP: $10^{12}$ Elemente, 9 Stunden, 1 TB." Die Motivation für die gesamte Krylov-Maschinerie steht dann direkt neben dem Spielzeug.

---

**4.2 · Probing-Matrix-Baukasten** ➕ · **M**

Auf demselben vorberechneten $\mathbf{T}$: Probing-Matrix $\Pi$ wählbar —
- Alles-Eins (= gewöhnliches Bild)
- Diagonale (direkter Anteil)
- $w$-te Nebendiagonale (lokale Streuung, mit $w$-Regler)
- Bernoulli mit Reglern für $p$ und $K$
- $\mathbf{m}_k = \mathbf{1} - \mathbf{p}_k$ (nur indirekt)
- Differenz zweier Probings

Angezeigt: die approximierte $10\times10$-Probing-Matrix (wie Ihre Beispielbilder), das resultierende Bild, und der Verstärkungsfaktor $1/p$ als Zahl. Beim Verkleinern von $p$ sieht man Kontrastgewinn *und* Rauschen zunehmen — der Trade-off, der im Skript nur implizit ist.

Als Szene bietet sich Ihr Testchart in trüber Milchlösung an, weil die Wirkung dort drastisch ist.

---

**4.3 · Optische Power-Iteration / Arnoldi als Schrittfolge** ➕ · **M**

„Nächste Iteration"-Knopf. Pro Schritt gezeigt: projiziertes Muster $\mathbf{p}_k$ (bzw. $\mathbf{p}_k^+$ und $\mathbf{p}_k^-$ bei Arnoldi), aufgenommenes Bild $\mathbf{i}_k$, normiertes $\mathbf{p}_{k+1}$ — plus rechts eine Konvergenzkurve und die aktuelle Rang-$K$-Approximation von $\mathbf{T}$ im Vergleich zum Original.

Der Regler „$K$" beantwortet die Frage, die im Skript offen bleibt: *Wie viele Iterationen braucht man eigentlich?* Bei $K = 5$ ist die Szene schon erstaunlich gut relightbar — das ist die Pointe der ganzen Krylov-Argumentation.

**Wettbewerbsvariante:** „Wer schafft ein erkennbares Relighting mit dem kleinsten $K$?" (siehe Abschnitt 6).

---

### Kapitel 5 — Neural Networks

Hier gibt es viel gutes externes Material (TensorFlow Playground, distill.pub, 3Blue1Brown). Eigene Widgets lohnen deshalb nur dort, wo Ihre Darstellung von der Standarddarstellung abweicht — und das ist beim Approximationstheorem und beim Backprop-Graphen der Fall.

---

**5.1 · Universeller-Approximator-Baukasten** ➕ · **M** · ⭐

Direkt aus Ihren `plot_2_neurons(s1,s2,w1,w2)` und `plot_2_rects(...)` entwickelt, aber auf $N$ Neuronenpaare erweitert:

- Zielfunktion vorgeben **oder mit der Maus zeichnen**
- $N$-Regler (1 … 30 Rechteckbausteine)
- Studierende schieben $s_i$ und $h_i$ von Hand → live berechneter Approximationsfehler
- **Dann der Knopf „Gradientenabstieg starten"** → dieselben Parameter werden automatisch optimiert, mit Lernraten-Regler und Verlustkurve

Der didaktische Clou liegt in der Verbindung: Ihr Skript behandelt Approximationstheorem (Abschnitt „Universal approximation theorem") und Optimierung (Abschnitt „Optimization of network parameters") getrennt. Im Widget ist es **derselbe Parametersatz**, einmal von Hand und einmal automatisch. Der Satz „das Netz lernt genau diese Parameter" braucht danach keine weitere Erklärung.

Zusatz-Toggle „mehr, aber schmalere Schichten": zeigt Ihre Schlussbemerkung, dass tiefe Netze mit weniger Bausteinen auskommen.

---

**5.2 · Layer-Visualizer (Conv / Transposed Conv / Pooling)** 🔁➕ · **M**

Ein Widget für drei Ihrer Bilderfolgen (`convolutional_layer_`, `transposed_convolutional_layer_`, `pooling_layer_`):

- Eingangsbild klein und wählbar, Kernel als **editierbares 3×3-Zahlenfeld**
- Regler für Stride, Padding-Modus, Kernelgröße
- animiertes Fenster, das über die Eingabe wandert, mit live gefüllter Ausgabe
- **Transposed-Modus:** man sieht die additive Überlagerung entstehen — genau der Punkt, den Sie im Text betonen („In that case, the new values are additively superposed") und der aus Standbildern schwer wird
- Presets: Identität, Sobel, Gauß, Zufall

Ein Ausgabegrößen-Rechner („$\lfloor (N + 2p - k)/s \rfloor + 1$") daneben, der live mitläuft, erspart in der Übung erfahrungsgemäß viel Verwirrung.

---

**5.3 · Backpropagation-Rechengraph zum Durchklicken** ➕ · **M** · ⭐

Ihr Beispiel $y = \log(x)^2$ mit den Zwischenvariablen $v_1, v_2$ als gezeichneter Graph.

1. **Forward-Pass:** $x$ eingeben, Knoten füllen sich nacheinander mit Zahlen
2. **Backward-Pass:** Knoten für Knoten anklicken → lokale Ableitung erscheint, das akkumulierte Produkt läuft rückwärts durch den Graphen
3. Vergleich mit `torch.autograd`-Ausgabe daneben

Ausdruck erweiterbar auf ein paar Varianten ($\sin(x^2)$, $e^{-x^2}$, ein Zweineuronen-Netz mit Verzweigung — wichtig, weil dort **Gradienten aufsummiert** werden, was im linearen Beispiel nicht sichtbar wird).

*Ideal für „erst raten lassen, dann aufdecken":* Sie zeigen den Graphen, fragen „Was steht gleich an dieser Kante?", lassen abstimmen, dann klicken Sie.

---

**5.4 · Aktivierungsfunktions-Vergleicher** 🔁 · **S**

Alle sechs Funktionen (Sigmoid, ReLU, Leaky ReLU, ELU, Softplus, tanh) überlagerbar, mit Reglern für $\beta$ und Leaky-Slope. **Entscheidende Ergänzung: die Ableitung zuschaltbar** — dann sieht man das Verschwinden des Sigmoid-Gradienten in den Flanken sofort, und die Frage „warum eigentlich ReLU?" beantwortet sich selbst.

---

### Kapitel 6 — Inverse Problems

Vier Vorlesungstermine, das mit Abstand längste und formalste Kapitel (3200 Zeilen im Textexport). Gleichzeitig das Kapitel, in dem am meisten *Verfahren nebeneinander* stehen, ohne dass die Studierenden sie je unter identischen Bedingungen verglichen sehen.

---

**6.1 · Die Dekonvolutions-Werkbank** ✨ · **L** · ⭐⭐ *das wichtigste Einzelwidget der Vorlesung*

Ein Widget, das das gesamte Kapitel zusammenhält.

**Oben — das Vorwärtsmodell:**
- Testbild wählbar
- PSF $h$ wählbar: Defokus-Pillbox (Radius-Regler), Bewegungs-Rect (Länge und Winkel), codiertes Rect (Raskar!), Gauß
- Rauschpegel $\sigma$-Regler, Rauschart Gauß/Poisson umschaltbar
- angezeigt: $g = s * h + n$

**Unten — Reiter mit den Verfahren, alle auf demselben $g$:**

| Reiter | Regler | Was sichtbar wird |
|---|---|---|
| Inversfilter | — | explodiert schon bei winzigem $\sigma$ |
| Wiener | SNR-Modell: Heuristik $1/\|f\|^2$ (Exponent-Regler) / konstant / **Ground Truth** / Deep-SNR (vorberechnet) | Ihre Tabelle PSNR 32,2 → 40,8 → 42,3 wird selbst erzeugt |
| Richardson–Lucy | **Iterationszahl-Regler** | PSNR steigt, erreicht Maximum, **fällt wieder** |
| HQS | $\lambda$, $\rho$, Iterationen, Prior-Auswahl | Prior-Wechsel als Plug-and-play |
| ADMM | dito + $\mathbf{u}$-Update sichtbar | konvergiert dort, wo HQS stagniert |

**Permanent eingeblendet:** PSNR und SSIM zum Ground Truth, plus die MTF der aktuellen PSF (Wiederverwendung von Widget 2.1).

**Warum das so viel wert ist:** Ihr Kapitel enthält sechs Verfahren, die im Skript in sechs getrennten Abschnitten mit sechs getrennten Beispielbildreihen (`inv_filter_res_`, `wiener_filter_res_`, `dw_ex_` …) vorkommen. Die Frage, die in der mündlichen Prüfung mit Sicherheit kommt — *„Wann würden Sie was nehmen und warum?"* — beantwortet dieses Widget in dreißig Sekunden Reglerbewegung.

**Zwei besonders starke Demo-Momente:**
- *Inversfilter:* $\sigma$ von exakt 0 minimal hochziehen. Der Zusammenbruch ist so abrupt, dass er im Hörsaal ein Geräusch erzeugt. Vorher abstimmen lassen: „Bei welchem $\sigma$ bricht es zusammen?"
- *Richardson–Lucy:* den Iterationsregler über das Optimum hinausziehen. Ihre Warnung im Text („starts to amplify noise after its usually quick convergence") wird zu einer sichtbaren Kurve mit Maximum — und zur naheliegenden Anschlussfrage „Wie würden Sie das automatisch stoppen?"

*Umsetzungshinweis:* Für den ersten Wurf reichen vorberechnete Kacheln (Python offline über das Parameterraster, PNG-Export, im Widget nur durchblenden). Live-Rechnung nur für Inversfilter und Wiener — die sind billige FFT-Operationen.

---

**6.2 · Regularisierungs-Explorer / Prior-Vergleich** ✨ · **M** · ⭐

Ein 1D-Signal (Stufenfunktion mit Rauschen) **und** ein kleines Bild, beide gleichzeitig.

Regularisierer wählbar, exakt Ihre Liste aus dem Skript:
- $\Psi(s) = \|\Delta s\|_2^2$ — Glattheit, für unscharfe Bilder
- $\Psi(s) = \|s\|_1$ — Sparsity, für Sternenfelder
- $\text{TV}$ anisotrop / isotrop — sparse Gradienten, für natürliche Bilder
- CNN-Denoiser (vorberechnet) — der Plug-and-play-Fall

$\lambda$-Regler von 0 bis stark.

Was man dann sieht und *nicht* wegdiskutieren kann: Tikhonov verschmiert Kanten, TV erzeugt bei großem $\lambda$ den charakteristischen Treppenstufeneffekt (Cartoon-Artefakte), $\ell_1$ funktioniert brillant auf dem Sternenfeld und katastrophal auf dem Porträt. Ihre Zuordnungsliste „Blurry images → smoothness, sparse images → sparsity, natural images → sparse gradients" wird von einer Merkregel zu einer Beobachtung.

*Passende Aktivierung:* drei Testbilder (Sterne, Porträt, Strichzeichnung), drei Priors, Studierende ordnen zu, **bevor** Sie klicken.

---

**6.3 · Soft-Thresholding, skalar und vektoriell** 🔁➕ · **S**

Links Ihr bestehendes `soft_thres_visu(v)`: die Funktion $b(z)$ mit markiertem Minimum, daneben die Kennlinie $S_{\lambda/\rho}(v)$.

Rechts neu: **vektorielles Soft-Thresholding** für den isotropen TV-Fall. 2D-Vektor $\mathbf{v}' = (v_i, v_{i+N})^\intercal$ mit der Maus ziehbar, der „Totkreis" mit Radius $\lambda/\rho$ eingezeichnet. Innerhalb → Ergebnis null. Außerhalb → radial verkürzt.

Der Unterschied zwischen anisotrop (Quadrat, achsenweise) und isotrop (Kreis, radial) ist genau die Sorte Detail, die auf Papier untergeht und in einem Bild sofort klar ist. Ihre Herleitung dazu ist lang und formal — das Bild ist ein Kreis.

---

**6.4 · Proximal-Operator-Geometrie** ➕ · **M**

2D-Konturplot einer konvexen Funktion $f$ mit sichtbarer Domäne. Punkt $\mathbf{v}$ mit der Maus ziehen, $\text{prox}_{f,\lambda}(\mathbf{v})$ wird als zweiter Punkt mit Verbindungspfeil gezeichnet. $\lambda$-Regler.

Genau Ihre Skript-Abbildung, aber mit dem Punkt in der Hand des Betrachters. Die beiden Aussagen — Punkte innerhalb wandern Richtung Minimum, Punkte außerhalb auf den Rand — werden durch Herumziehen zu etwas, das man ausprobiert hat. Zusatz-Toggle: „$\text{prox}_{f,\lambda}(\mathbf{v}) \approx \mathbf{v} - \lambda\nabla f(\mathbf{v})$ für kleines $\lambda$" zeigt beide Punkte übereinander, und man sieht, ab wann die Näherung bricht.

---

**6.5 · HQS/ADMM-Schrittdurchlauf mit Konvergenzvergleich** ✨ · **M**

Einzelne Updates einzeln steppen: $s$-Update → $z$-Update → ($u$-Update). Nach jedem Schritt: aktuelles Bild, Residuum $\|\mathbf{Ds}-\mathbf{z}\|$, Zielfunktionswert.

**Der entscheidende Modus:** ein Umschalter zwischen zwei Problemen —
- Dekonvolution ($M = N$): HQS und ADMM konvergieren beide gut
- Inpainting / Compressed Sensing mit 10 % Messungen ($M \ll N$): **HQS stagniert, ADMM konvergiert**

Damit wird Ihr Abschnitt „Problems of HQS" von einer Bemerkung am Kapitelende zu einer Demonstration. Das ist die Art Beobachtung, die in der mündlichen Prüfung als eigene Erinnerung abrufbar ist statt als auswendig gelernter Satz.

---

**6.6 · Schlechtgestelltheit im Kleinen** ✨ · **S**

Ein $3\times3$-Gleichungssystem $\mathbf{g} = \mathbf{H}\mathbf{s}$ mit Regler für die Konditionszahl von $\mathbf{H}$. $\mathbf{g}$ mit winzigem Rauschen stören → $\hat{\mathbf{s}} = \mathbf{H}^{-1}\mathbf{g}$ explodiert. Singulärwerte als Balken daneben, Tikhonov-$\lambda$ zuschaltbar → man sieht, wie die kleinen Singulärwerte gedämpft werden.

Winziges Widget, aber es macht Ihre drei Hadamard-Kriterien (keine Lösung / keine eindeutige Lösung / keine stetige Abhängigkeit) an einem Objekt konkret, das man vollständig überblickt — bevor es um Millionen Bildpixel geht. Guter Einstieg in den ersten der vier Termine.

---

**6.7 · Deep-SNR gegen Heuristik** 🔁 · **S**

Drei SNR-Kurven überlagert: Ground Truth, Heuristik $1/\|f\|^2$, netzgeschätzt. Darunter die drei resultierenden Wiener-Filter und die drei Rekonstruktionen, umschaltbar.

Vorberechnet, deshalb billig. Ihre Tabelle (32,23 → 40,83 → 42,31 dB) bekommt damit ein Bild, und die Aussage „das Netz kommt fast an den Ground Truth heran" wird sichtbar statt behauptet.

---

### Kapitel 7 — Lensless Imaging

---

**7.1 · DiffuserCam-Simulator** ✨ · **L** · ⭐

**Vorwärtsrichtung:** Szene (bewegliche helle Punkte oder ein kleines Bild) → Faltung mit einer echten Kaustik-PSF → Crop auf Sensorgröße → Rauschen. Alles live.

Regler: Punktposition (lateral und axial → PSF verschiebt bzw. skaliert sich), Sensor-Crop, Rauschpegel.

**Modulator-Umschalter** — dieser Toggle ist der wichtigste Teil:
- offener Sensor (keine Modulation) → alle Szenenpunkte erzeugen fast dieselbe Antwort → Rekonstruktion unmöglich
- Lochblende → funktioniert, aber fast kein Licht
- Amplitudenmaske → mittelmäßiges SNR
- Diffusor/Kaustik → viel Licht, gute Rekonstruktion

Damit wird Ihr Satz „Just using a sensor without any modulator results in a severely ill-posed problem" zu etwas, das man im Rekonstruktionsergebnis scheitern sieht.

**Rückwärtsrichtung:** projizierter Gradientenabstieg mit Iterationsregler (bildet `diffuser_cam_grad_recon_`, 10 Stufen, ab) und ADMM (bildet `diffuser_cam_ADMM_recon_`, 10 Stufen, ab), nebeneinander mit Iterationszähler. Der Geschwindigkeitsunterschied ist die Pointe des Abschnitts.

*Anschluss:* Übung „Diffuser Cam" am 18.02. Das Widget ist die perfekte Vorbereitung — und umgekehrt können Studierende ihre eigene Implementierung gegen das Widget prüfen.

---

**7.2 · Verschiebungsinvarianz-Test** 🔁➕ · **S**

Punktquelle mit der Maus im Sichtfeld bewegen → PSF wandert (lateral) bzw. skaliert (axial). Jenseits eines einstellbaren FOV-Winkels **bricht die Verschiebungsinvarianz sichtbar zusammen** (die PSF verzerrt sich), und ein Warnhinweis erscheint: „Faltungsmodell nicht mehr gültig → lokale Faltungsmodelle."

Ersetzt `diffuser_cam_shift_invaraince_` (7 Stufen) und macht Ihre drei Gültigkeitsbedingungen (schmales FOV, ausreichender Abstand, Fernfeld) prüfbar statt aufzählbar.

---

**7.3 · Fresnel-Zahl-Rechner** ✨ · **S**

Regler $a$ (kleinste Maskenöffnung), $d$ (Masken-Sensor-Abstand), $\lambda$ → $N_F = a^2/(d\lambda)$ als große Zahl, farbcodiert. Daneben qualitativ das Sensorbild: bei $N_F \gg 1$ ein scharfer Schatten, bei $N_F \le 1$ ein Beugungsmuster, dazwischen der Übergang.

Klein, aber es macht eine Größenordnungsabschätzung zu einer Erfahrung — und Größenordnungsabschätzungen sind genau das, was in der Praxis über die Wahl des Modells entscheidet.

---

**7.4 · Separabilitäts-Demo** ✨ · **S**

$\Phi_L$ und $\Phi_R$ als zwei Vektoren mit der Maus zeichenbar → die Maske $\Phi_L\Phi_R^\intercal$ entsteht als äußeres Produkt. Danach der Versuch, ein Zielmuster (z. B. eine Kaustik oder eine MURA-Maske) durch Zeichnen zu treffen — **es geht nicht**.

Damit wird spürbar, wie stark die Separabilitätsannahme die Modellklasse einschränkt, und warum sie trotzdem attraktiv ist (Speicher- und Rechenaufwand als Zahl daneben).

---

### Kapitel 8 — Coded Exposure Photography

Das kürzeste Kapitel (196 Zeilen) mit dem — meiner Einschätzung nach — höchsten Aktivierungspotenzial pro investierter Minute im ganzen Semester.

---

**8.1 · Fluttering-Shutter-Code-Designer** ✨ · **M** · ⭐⭐ *stärkstes Aktivierungswidget*

Der 52-Bit-Code als **klickbare Bit-Leiste**. Die Studierenden schalten einzelne Bits um.

Live daneben, alles gleichzeitig:
- Log-Magnitudenspektrum des Codes, mit **Nullstellenzähler** und **Minimalwert**
- Autokorrelation (Ziel: dirac-ähnlich)
- Lichteffizienz = Anzahl der Einsen / 52
- verwackeltes Testbild mit genau diesem Code
- Wiener-Rekonstruktion davon, mit RMSE zum Original

Presets: Box (alle 1 = konventionelle Belichtung), **Raskar-Code** `1010000111000001010000110011110111010111001001100111`, Zufall, „alle 52 zufällig neu würfeln".

**Die Vorlesungsaktivität (10 Minuten, Zweiergruppen):**

> „Raskar et al. haben für $M = 52$ eine randomisierte lineare Suche laufen lassen. Sie haben jetzt fünf Minuten und Ihre Finger. Ziel: kleineres RMSE bei mindestens 50 % Lichteffizienz. Schicken Sie mir Ihre URL."

Das funktioniert aus mehreren Gründen ausgesprochen gut:
- Der Suchraum ist riesig ($\binom{52}{26} \approx 5\cdot10^{14}$), aber die Zielfunktion ist sofort sichtbar → echtes Optimierungsgefühl
- Der Zielkonflikt (Invertierbarkeit ↔ Lichteffizienz) wird körperlich: jedes zusätzliche Eins-Bit hilft dem Licht und kann dem Spektrum schaden
- **Fast niemand schlägt Raskar** — und genau das ist die Lehre: „Deshalb macht man das nicht von Hand. Deshalb machen wir in Kapitel 12 Ende-zu-Ende-Optimierung."
- Es lässt sich mit URL-Parametern als Leaderboard führen

Der Bogen von Kapitel 8 (Handsuche) über die randomisierte Suche im Paper zu Kapitel 12 (gradientenbasierte Ende-zu-Ende-Optimierung) ist der roteste Faden, den die Vorlesung zu bieten hat. Dieses Widget spannt ihn.

---

**8.2 · Bewegungsunschärfe-Simulator** ➕ · **S**

Objekt mit einstellbarer Geschwindigkeit und Richtung vor Hintergrund, Belichtungszeit $T$-Regler → Verwischung entsteht als Animation, mit eingeblendetem Kernel $h$ als rotiertes Rect.

Vor allem als *Vorbereitung* auf 8.1 nützlich: erst verstehen, woher das Rect kommt und warum seine Fourier-Transformierte eine sinc mit Nullstellen ist, dann den Code entwerfen. Der Verweis auf Widget 2.1 (PSF↔MTF) schließt sich hier.

---

### Kapitel 10 — Coded Aperture Spectral Snapshot Imaging

Kurzes Kapitel (280 Zeilen), aber mit vier Architekturen, die sich nur durch die Reihenfolge derselben drei Operatoren unterscheiden — geradezu prädestiniert für ein Baukastenwidget.

---

**10.1 · CASSI-Baukasten** ✨ · **L** · ⭐

Ein kleiner echter Hyperspektralwürfel ($32\times32\times8$ reicht völlig) als Eingang. Darunter eine **Pipeline aus zu- und abschaltbaren Bausteinen**:

`Eingang x` → `diag(p) Maske` → `T Scherung` → `Tᵀ inverse Scherung` → `Σ Projektion` → `Messung y`

Die Maske $\mathbf{p}$ ist zeichenbar. Jeder Zwischenschritt wird als $(x,\lambda)$-Diagramm dargestellt — exakt Ihre Abbildungen des Abtastschemas, aber als lebende Zwischenzustände.

**Architektur-Presets, die die Bausteine automatisch richtig setzen:**

| Preset | Kette | Ortsauflösung | Rekonstruktion nötig? |
|---|---|---|---|
| RGB-Bayer | $\Sigma\,\text{diag}(\mathbf{p})$ | voll | nein |
| PMVIS | $\Sigma\mathbf{T}\,\text{diag}(\mathbf{p})$ | reduziert | nein |
| SD-CASSI | $\Sigma\mathbf{T}\,\text{diag}(\mathbf{p})$ | voll | **ja** |
| DD-CASSI | $\Sigma\mathbf{T}^\intercal\text{diag}(\mathbf{p})\mathbf{T}$ | voll | ja |

Der Unterschied zwischen PMVIS und SD-CASSI liegt nur in der Maskendichte — dass sie dasselbe Vorwärtsmodell $\mathbf{y} = \mathbf{A}\mathbf{x}$ haben und trotzdem völlig verschiedene Rekonstruktionsanforderungen, ist genau das, was man am Widget sofort sieht und aus dem Text nur mühsam herausliest.

Rechts: TV-regularisierte Rekonstruktion (klein genug für Live-Rechnung) mit spektralem Profil an einer anklickbaren Bildposition.

---

**10.2 · Datenraten-Rechner** ✨ · **S**

Regler: Spektralbänder $S$, Ortsauflösung $N$, Bildrate, Bittiefe → Datenrate in GB/s, groß dargestellt. Daneben die CASSI-Messgröße $N + S - 1$ und der Kompressionsfaktor.

Ihr Satz „ein einziges Sekunde unkomprimiertes Hyperspektralvideo mit 60 Bändern und 1 MP ergibt etwa 2 GB" ist eine gute Zahl — als Regler wird daraus ein Gefühl dafür, wie schnell das eskaliert. Zehn Minuten Arbeit.

---

### Kapitel 11 — Time-of-Flight Imaging

---

**11.1 · Transient-Imaging-Scrubber** ➕ · **M** · ⭐

Zeitschieber $t$ (in Pikosekunden!) → der Lichtpuls wandert sichtbar durch eine 2D-Szene. Direkte und indirekte Pfade unterschiedlich eingefärbt. Rechts: das Histogramm $h(t)$ eines anklickbaren Pixels, mit Marker auf der aktuellen Zeit.

Ihr `transient_imaging_` hat 6 Stufen; als flüssiger Scrubber ist der Effekt ungleich stärker, weil man den Puls *wandern* sieht statt sechs Momentaufnahmen. Und der Kernsatz — „gewöhnliche Kameras integrieren über all diese transienten Bilder" — bekommt einen Knopf: „integrieren" blendet alle Zeitschritte zusammen, und man erhält das langweilige normale Foto.

Auf dieselbe Szene lässt sich die Direkt/Indirekt-Trennung aus Kapitel 4 legen — dieselbe Unterscheidung, einmal über Probing, einmal über Laufzeit. Schöner Querverweis für die letzte Vorlesung.

---

**11.2 · SPAD-Histogramm-Simulator** ✨ · **M**

Regler: Anzahl Pulse $N$ (1 bis $10^7$, logarithmisch), Quanteneffizienz $\eta$, Totzeit, Dunkelzählrate $d$, Jitter $f$.

Das Histogramm baut sich **sichtbar auf** (animiert), mit Poisson-Rauschen. Bei $N = 100$ ist nichts erkennbar, bei $N = 10^6$ steht die Kurve.

Zwei Effekte, die nur so klar werden:
- **Warum Millionen Pulse:** Ihr Satz „usually millions of laser pulses are emitted" bekommt eine Begründung, die man am Rauschabfall $\propto 1/\sqrt{N}$ ablesen kann
- **Pile-up durch Totzeit:** bei hoher Photonenrate verzerrt sich das Histogramm zu frühen Zeiten hin — ein realer Messfehler, den man am Regler erzeugen kann

Ihre Formel $h(t) \sim \mathcal{P}(N\lambda(t))$ wird damit zur beobachteten Statistik.

---

**11.3 · NLOS-Lichtkegel** ✨ · **M** · ⭐

Draufsicht: Wand, Laser/SPAD-Punkt $(x', y')$ auf der Wand, versteckter Bereich dahinter.

- Punkt auf der Wand anklicken + Zeitregler $t$ → **die Kugelschale mit Radius $tc/2$** wird im versteckten Raum eingezeichnet: „irgendwo hier ist das Objekt"
- Zweiter, dritter Punkt → Schnittmengen → die Lokalisierung entsteht
- Toggle: Rückprojektion aller Abtastpunkte → das Objekt erscheint

Das ist die geometrische Intuition, die Ihre Formel

$$\tau(x',y',t) = \iiint \tfrac{1}{r^4}\,\rho(x,y,z)\,\delta\!\left(2\sqrt{(x'-x)^2+(y'-y)^2+z^2} - tc\right)dx\,dy\,dz$$

trägt: **die Delta-Funktion ist genau diese Kugelschale.** Wer das einmal gesehen hat, liest die Formel danach anders.

`nlos_task_` hat 7 Stufen — als interaktive Version wird daraus etwas, mit dem man spielt.

---

### Kapitel 12 — End-to-End Optimization

Das Kapitel mit den kaputtesten Interaktionen (fünf JS-Injection-Stepper, die im PDF „JavaScript output is disabled" anzeigen) und gleichzeitig das mit den bereits eingebauten `❓ Question`-Boxen — das Aktivierungsformat ist hier also schon da und muss nur bedient werden.

---

**12.1 · Ende-zu-Ende-Sandkasten: Mensch gegen Optimierer** ✨ · **L** · ⭐⭐

Ein kleines, vollständiges Ende-zu-Ende-System:

`PSF-Maske (8×8 binär, klickbar)` → `Faltung` → `Rauschen` → `ADMM (wenige Iterationen)` → `Rekonstruktion` → `Loss`

Die Studierenden malen eine Maske von Hand und sehen sofort: Loss-Wert, MTF der Maske, Lichteffizienz $\frac{1}{|\Omega|}\sum_x h(x)$, Rekonstruktionsqualität.

**Dann der Knopf „Optimieren".** Gradientenabstieg läuft (vorberechnet als Frame-Sequenz genügt), und man sieht die Maske sich entwickeln — daneben die Loss-Kurve.

Zusätzliche Regler, die direkt Ihre `❓ Question`-Boxen bedienen:
- **Lichteffizienz-Ziel $p$** → Ihr $\ell_{\text{PSF}}(h) = \left(\frac{\sum_x h(x)}{|\Omega|} - p\right)^2$ wird als zweite Kurve mitgeführt. Man sieht den Zielkonflikt: rekonstruktionsoptimal vs. lichteffizient.
- **Binarisierungsschärfe $\gamma$** → Histogramm der PSF-Koeffizienten wandert bei steigendem $\gamma$ zu den Rändern 0 und 1. Ihre Frage „What could be done to force the values to be close to either 0 or 1?" wird beantwortet, indem man den Regler zieht.

Das ist der Höhepunkt der Vorlesung: der Bogen von Kapitel 8 (Code von Hand) über Kapitel 6 (ADMM als differenzierbare Kette) zu Kapitel 12 (alles gemeinsam optimiert) schließt sich in einem Widget.

---

**12.2 · Optisches Zero-Padding** ✨ · **M**

Zwei gekoppelte Panels.

**Links, Geometrie:** Regler $\Delta$, $f$, $d_L$, $d_s$ → gezeichneter Strahlengang mit der berechneten Stufengröße
$$d_f = 2\left(\tfrac{(d_L+d_s)/2}{f+\Delta}\cdot f - \tfrac{d_L}{2}\right)$$

**Rechts, Konsequenz:** Rekonstruktion mit und ohne Zero-Padding. Ohne: die Wrap-around-Artefakte am Bildrand, weil die FFT-Dekonvolution zyklische Faltung annimmt, die Optik aber nicht zyklisch faltet.

Ihre `❓ Question` an dieser Stelle — *„Angenommen, Sie haben nur einen Faltungsoperator ohne Wrap-around. Wie sorgen Sie trotzdem für ein zyklisches Ergebnis?"* — bekommt damit erst das Problem (sichtbare Artefakte), dann die Denkzeit, dann die Lösung. Genau die richtige Reihenfolge.

Das ist außerdem einer der wenigen Punkte im Skript, an dem eine *rein optische* Maßnahme ein *rein numerisches* Problem löst — für die Kernbotschaft der Vorlesung („Optik und Algorithmus gemeinsam denken") ein Kronzeuge.

---

**12.3 · Deflektometrie-Musteroptimierer** ✨ · **M**

Bildschirmmuster wählbar: Streifen (Periode und Phase einstellbar — die klassische heuristische Wahl), Zufall, **gelerntes Muster** (Ihr Optimierungsergebnis). Synthetisches Prüfobjekt mit zufälligen Defekten, Kamerabild, Defektkontrast als Zahl.

Ihr Skript sagt an dieser Stelle: „⚡ Screen patterns are not optimized but chosen heuristically! 💡 Idea: Optimize pattern in end-2-end manner via learning." — Mit dem Widget dürfen die Studierenden erst selbst heuristisch optimieren (Streifenperiode variieren, bis der Kontrast maximal ist), bevor das gelernte Muster gezeigt wird. Der Vergleich wirkt dann.

---

## 4. Kapitelübergreifende Widgets

Diese drei sind keiner Vorlesung zugeordnet, sondern verbinden mehrere — und sind deshalb besonders wertvoll für die Prüfungsvorbereitung.

---

**Ü1 · Der rote Faden: „Ein Modell, viele Modalitäten"** ✨ · **M** · ⭐

Ein Widget, in dem man die Modalität wählt und **immer dieselbe Gleichung** $\mathbf{g} = \mathbf{H}\mathbf{s} + \mathbf{n}$ sieht — nur mit anderem $\mathbf{H}$:

| Modalität | $\mathbf{H}$ ist … | Kapitel |
|---|---|---|
| Defokus | zirkulante Toeplitz-Matrix (Pillbox-Faltung) | 2, 6 |
| Bewegungsunschärfe | zirkulante Toeplitz (Rect / codiertes Rect) | 8 |
| Lensless | Faltung mit Kaustik + Crop $\mathbf{C}$ | 7 |
| Lichttransport | die volle Transportmatrix $\mathbf{T}$ | 4 |
| CASSI | $\Sigma\mathbf{T}\,\text{diag}(\mathbf{p})$ | 10 |
| NLOS | Lichtkegel-Transformation | 11 |
| Deflektometrie | Musterabbildung | 3, 12 |

Für jede: $\mathbf{H}$ als Bild (bzw. als Struktur-Schema), die Messung $\mathbf{g}$, und dieselbe ADMM-Rekonstruktion mit denselben Reglern $\lambda, \rho$.

**Einsatz:** letzte Vorlesung am 18.02. als Zusammenfassung. Die Botschaft — *Computational Imaging ist im Kern ein Fach, nicht zwölf* — lässt sich kaum stärker transportieren als dadurch, dass derselbe Löser auf sieben völlig verschiedenen Geräten funktioniert. Das ist auch die Struktur, an der man sich in der mündlichen Prüfung entlanghangeln kann, wenn einem gerade nichts einfällt.

---

**Ü2 · Design-Space-Navigator** ✨ · **S**

Ihre Freiheitsgrade aus Kapitel 1 (Beleuchtung: Intensität, Richtung, Profil, Spektrum, Polarisation, Muster · Bildaufnahme: Linse, Brennweite, Verschluss, Blende, Filter, Integrationszeit) als anklickbare Matrix.

Klick auf eine Kombination → welche Verfahren der Vorlesung nutzen sie? Klick auf ein Verfahren → welche Freiheitsgrade nutzt es?

Beispiel: „programmierbare Blende + Integrationszeit" → Coded Exposure. „Beleuchtungsmuster + Kameramodulation" → Primal-Dual-Coding. „Richtung + Muster" → Lichtfeldbeleuchtung.

Klein, aber es gibt der Vorlesung eine Landkarte — einsetzbar in Woche 1 als Ausblick und in Woche 15 als Rückblick, mit den zwischenzeitlich gefüllten Feldern.

---

**Ü3 · Notations-Nachschlagewerk** ✨ · **S**

Eine durchsuchbare Tabelle aller Symbole mit Kapitelverweis: $L$, $\rho$ (Plenoptik!) vs. $\rho$ (ADMM-Gewicht!) vs. $\rho$ (Albedo!), $\mathbf{T}$, $\mathbf{H}$, $\Psi$ (Regularisierer) vs. $\Psi$ (Differenzoperator in Kap. 7), $\Pi$, $\mathbf{D}$, $\alpha$ (Refokussierung) vs. $\alpha$ (Ablenkwinkel) vs. $\alpha$ (ADMM-Gewicht), $\lambda$ (Wellenlänge!) vs. $\lambda$ (Regularisierungsgewicht) vs. $\lambda$ (Eigenwert) vs. $\lambda$ (Poisson-Erwartungswert).

Diese Mehrfachbelegungen sind in einer breit angelegten Vorlesung unvermeidbar und für Studierende eine echte Stolperfalle — besonders $\lambda$, das in Kapitel 6 als Regularisierungsgewicht und in Kapitel 7/10 als Wellenlänge auftritt. Ein Klick-Nachschlagewerk kostet fast nichts und nimmt spürbar Reibung raus.

---

## 5. Priorisierung — was zuerst

### Stufe 0 — Sofort, hoher Ertrag

| # | Widget | Aufwand | Wirkung |
|---|---|---|---|
| 2.3 | Generischer Bilderfolgen-Stepper | S | **~25 tote Stellen im Web-Buch auf einmal wiederbelebt** |
| 2.1 | PSF ↔ MTF-Doppelansicht | M | trägt Kapitel 2, 6, 7, 8 **und** 12 |
| 8.1 | Coded-Exposure-Code-Designer | M | stärkste Aktivierung pro Minute |

Diese drei zusammen sind realistisch ein bis zwei Wochen Arbeit und verändern das Erlebnis der Vorlesung bereits deutlich.

### Stufe 1 — Größte inhaltliche Lücken

| # | Widget | Aufwand | Wirkung |
|---|---|---|---|
| 6.1 | Dekonvolutions-Werkbank | L | hält vier Vorlesungstermine zusammen |
| 3.2 | $(x,u)$-Lichtfeld-Diagramm | M | schließt die größte Verständnislücke in Kap. 3 |
| 3.1 | Lichtfeld-Explorer | L | das Widget mit dem höchsten Wow-Faktor |
| 12.1 | Ende-zu-Ende-Sandkasten | L | Höhepunkt und Klammer der Vorlesung |
| 4.1 | Lichttransportmatrix-Sandkasten | L | macht das abstrakteste Kapitel greifbar |

### Stufe 2 — Solide Ergänzungen

6.2 Prior-Explorer · 6.5 HQS/ADMM-Vergleich · 7.1 DiffuserCam · 11.3 NLOS-Lichtkegel · 10.1 CASSI-Baukasten · 5.1 Approximator · 5.3 Backprop-Graph · 3.4 Deflection-Map-Explorer · Ü1 Roter Faden

### Stufe 3 — Kleine, billige Gewinne (jeweils S)

2.6 Telezentrie · 2.7 Bokeh · 2.8 Dirac/Fourier · 3.6 SAT-Spiel · 6.3 Soft-Thresholding · 6.6 Schlechtgestelltheit · 7.3 Fresnel-Zahl · 7.4 Separabilität · 10.2 Datenraten · Ü2 Design-Space · Ü3 Notation

Diese lassen sich gut als **HiWi-Aufgaben oder Bonusaufgaben** vergeben — jedes einzelne ist überschaubar, gut spezifizierbar und liefert etwas Vorzeigbares.

### Vorschlag zum Vorgehen

Nicht alles auf einmal. Ein tragfähiger Rhythmus wäre: **pro Semester zwei bis drei Widgets aus Stufe 0/1**, gebaut jeweils kurz vor dem betreffenden Termin, plus ein bis zwei Kleine als studentische Arbeiten. Nach drei Semestern ist die Vorlesung durchgehend interaktiv, ohne dass es je ein Großprojekt war.

Für Stufe-1-Widgets lohnt sich der Gedanke an **Bachelorarbeiten oder Praktika**: „Interaktive Web-Visualisierung der Lichtfeld-Refokussierung" ist eine saubere, abgeschlossene Aufgabe mit vorzeigbarem Ergebnis — und die Studierenden, die sie bearbeiten, verstehen das Kapitel danach besser als alle anderen.

---

## 6. Aktivierung der Studierenden während der Vorlesung

### 6.0 Ausgangslage

Ihr Format — mittwochs 14:00–15:30 Vorlesung, jede zweite Woche 15:45–17:15 Übung, mündliche Prüfung — hat zwei Eigenschaften, die für Aktivierung günstig sind:

1. **90 Minuten am frühen Nachmittag** sind aufmerksamkeitstechnisch anspruchsvoll. Alle 15–20 Minuten ein Wechsel der Sozialform ist hier keine Spielerei, sondern Notwendigkeit.
2. **Mündliche Prüfung** heißt: Studierende müssen *sprechen* können, nicht nur rechnen. Jede Vorlesungsminute, in der sie Fachsprache produzieren statt konsumieren, ist direkte Prüfungsvorbereitung. Das ist ein starkes Argument, das Sie den Studierenden auch offen sagen sollten — es erhöht die Bereitschaft mitzumachen erheblich.

Außerdem: Sie haben in Kapitel 12 bereits `❓ Question`-Boxen im Skript. Das Format existiert also schon und muss nur systematisch ausgerollt werden.

---

### 6.1 Predict–Observe–Explain (das Grundritual zu jedem Widget)

Kein Widget ohne dieses Dreischritt-Ritual. Es dauert drei bis fünf Minuten und verwandelt eine Demonstration in Lernen:

1. **Predict** (60 s) — Frage stellen, Regler *nicht* bewegen. Abstimmung oder Handzeichen. „Was passiert mit dem Inversfilter, wenn ich $\sigma$ von 0 auf 0,001 erhöhe?"
2. **Observe** (30 s) — Regler ziehen. Schweigen.
3. **Explain** (2–3 min) — Erst Nachbargespräch, dann zwei bis drei Wortmeldungen. „Warum?"

Der entscheidende Teil ist Schritt 1. Eine Vorhersage, die falsch war, hinterlässt eine Gedächtnisspur; eine Demonstration ohne Vorhersage nicht. Es ist verlockend, den Regler direkt zu ziehen, weil es schneller geht — der Verzicht auf die Vorhersage kostet aber den Großteil des Effekts.

**Die zwölf besten Predict-Momente im Material:**

| Kapitel | Frage | Warum stark |
|---|---|---|
| 2 | „Ich blende ab. Was passiert mit Bokeh, Schärfentiefe, Helligkeit?" | drei Effekte gleichzeitig, mindestens einer wird falsch geraten |
| 2 | „MTF eines defokussierten Systems bei $r=3$ — wie viele Nullstellen?" | die meisten erwarten monotonen Abfall, nicht Nullstellen |
| 3 | „Ich ziehe $\alpha$. Wird das Vordergrund- oder Hintergrundobjekt zuerst scharf?" | erzwingt Nachdenken über das Vorzeichen der Steigung |
| 3 | „Wie ändert sich die Deflection Map, wenn ich über einen Streudefekt fahre?" | Peak-Zerfall ist überraschend drastisch |
| 4 | „$K=1$, $K=3$, $K=10$ Arnoldi-Iterationen — ab wann ist die Szene erkennbar?" | Schätzungen liegen fast immer zu hoch |
| 4 | „$p$ von 0,5 auf 0,05 — was gewinnen und was verlieren wir?" | zwingt zum Trade-off-Denken |
| 5 | „Wie viele Sigmoid-Paare brauche ich für *diese* Kurve?" | Approximationstheorem als Schätzaufgabe |
| 6 | „Inversfilter, $\sigma = 0{,}001$ — noch brauchbar?" | der Zusammenbruch ist spektakulär |
| 6 | „RL-Algorithmus: 5, 50, 500 Iterationen — welches ist am besten?" | „mehr ist besser" ist falsch |
| 7 | „Diffusor gegen offenen Sensor — welcher rekonstruiert besser?" | Intuition sagt oft „ohne Streuung besser" |
| 8 | „Box-Code gegen Raskar-Code bei gleicher Lichtmenge?" | siehe Wettbewerb unten |
| 11 | „Wie viele Laserpulse für ein brauchbares Histogramm?" | Größenordnungsgefühl |

---

### 6.2 Peer Instruction / ConcepTests

Das am besten belegte Format für große Vorlesungen (Mazur). Ablauf:

1. Konzeptfrage mit vier Antwortoptionen, Distraktoren sind typische Fehlvorstellungen
2. **Individuelle Abstimmung**, anonym
3. Bei 30–70 % richtig: **„Überzeugen Sie Ihre Nachbarin"** (2 Minuten)
4. **Zweite Abstimmung** — der Anteil richtiger Antworten steigt typischerweise deutlich
5. Auflösung durch Sie, mit Erklärung *warum die Distraktoren attraktiv sind*

**Tooling:** Am KIT ist Particify (Nachfolger von ARSnova) verfügbar und DSGVO-unbedenklich. Alternativen: tweedback, Mentimeter, oder — im kleinen Kurs am einfachsten — farbige Karten. Karten haben den Vorteil, dass Sie das Abstimmungsbild *sehen* und die Studierenden merken, dass Sie es sehen.

**Dosierung:** zwei bis drei ConcepTests pro 90 Minuten. Mehr ermüdet, weniger wird zur Ausnahme statt zum Ritual.

Konkrete Fragen: siehe Abschnitt 7.

---

### 6.3 Think–Pair–Share an Herleitungsstellen

Ihre längeren Herleitungen (HQS-Subdifferentiale, isotropes TV, CMD-Komplexität, Light-Cone-Transform) sind ausgezeichnete Stellen für einen Zwischenstopp — gerade *weil* sie lang sind.

**Rezept:** Herleitung bis zu einem Zwischenpunkt führen, dann stoppen:

> „Wir stehen bei $\rho(-v+z) + \partial\lambda|z| \overset{!}{=} 0$. Das Problem: $|\cdot|$ ist bei null nicht differenzierbar. **Zwei Minuten mit Ihrer Nachbarin: Was würden Sie tun?**"

Dann zwei Vorschläge einsammeln (auch und gerade die falschen — „glätten wir doch die Betragsfunktion" ist ein produktiver Vorschlag, den man kurz würdigen kann), dann das Subdifferential einführen. Der Begriff landet dann als *Antwort auf ein selbst empfundenes Problem* statt als weitere Definition.

**Gute Stopp-Punkte im Material:**

| Stelle | Frage an die Studierenden |
|---|---|
| Kap. 3, vor „shift and add" | „Die Integralformel steht da. Was fällt Ihnen an den Argumenten auf? Was wird eigentlich mit $(u,v)$ gemacht?" |
| Kap. 3, vor der CMD | „Wir brauchen eine Distanz zwischen zwei 2D-Verteilungen. Sammeln wir Anforderungen — was muss sie können?" |
| Kap. 4, vor Krylov | „Wir dürfen $\mathbf{T}$ nur mit Vektoren multiplizieren, nie ansehen. Was lässt sich daraus überhaupt lernen?" |
| Kap. 4, vor negativen $\mathbf{p}$ | „Krylov braucht Multiplikation mit beliebigen $\mathbf{p}$, auch negativen. Ein Projektor kann kein negatives Licht. Ideen?" |
| Kap. 6, vor HQS | „Direkter Gradientenabstieg auf $\frac12\|\mathbf{Hs}-\mathbf{g}\|^2 + \lambda\Psi(\mathbf{s})$ funktioniert schlecht. Warum wohl?" |
| Kap. 6, vor Soft-Thresholding | „$|\cdot|$ ist bei null nicht differenzierbar. Was nun?" |
| Kap. 6, vor ADMM | „HQS versagt bei stark unterbestimmten Problemen. Woran könnte das liegen?" |
| Kap. 7, Modulatorwahl | „Warum reicht ein nackter Sensor nicht? Formulieren Sie es mit $\mathbf{H}$." |
| Kap. 8, Codewahl | „Zwei widersprüchliche Ziele. Nennen Sie beide und sagen Sie, warum sie sich widersprechen." |
| Kap. 12, `❓`-Boxen | bereits im Skript — einfach benutzen! |

---

### 6.4 Wettbewerbsformate

Diese funktionieren, weil sie das Optimierungsproblem, über das Sie sprechen, in die Hände der Studierenden legen.

**W1 · Code-Design-Wettbewerb (Kap. 8, 10–15 min)** — siehe Widget 8.1.
Zweiergruppen, 5 Minuten Zeit, Ziel: RMSE unterbieten bei ≥ 50 % Lichteffizienz. Ergebnisse als URL einsammeln, Live-Leaderboard.
*Die Pointe:* fast niemand schlägt den Raskar-Code. Überleitung: „Deshalb Kapitel 12."

**W2 · Maskendesign-Duell (Kap. 12, 15 min)** — siehe Widget 12.1.
Gleiche Mechanik, aber danach läuft der Gradientenabstieg gegen die beste Handlösung an. Mensch gegen Optimierer, live.
*Die Pointe:* der Optimierer gewinnt, aber die Maske sieht *seltsam* aus. Anschlussfrage: „Warum sieht die so aus? Schauen Sie sich die MTF an." — führt direkt zu Ihrer bestehenden `❓`-Frage nach dem systemtheoretischen Einblick.

**W3 · Minimales $K$ (Kap. 4, 10 min)** — siehe Widget 4.3.
„Wer bekommt ein erkennbares Relighting mit den wenigsten Arnoldi-Iterationen?" Das Kriterium „erkennbar" ist bewusst unscharf und darf diskutiert werden — das ist Teil der Übung, weil es die Frage aufwirft, was Rekonstruktionsqualität eigentlich ist.

**W4 · Prior-Zuordnung (Kap. 6, 8 min)** — siehe Widget 6.2.
Drei Bilder (Sternenfeld, Porträt, technische Strichzeichnung), fünf Prior-Optionen. Gruppen ordnen zu und begründen, dann wird geklickt. Die Begründung zählt mehr als die Zuordnung.

**W5 · Blindverkostung (Kap. 6, 5 min)**
Vier Rekonstruktionen desselben verrauschten Bildes, unbeschriftet. „Welche stammt vom Wiener-Filter mit Heuristik, welche vom Deep-SNR, welche von RL, welche von ADMM mit TV?" Die charakteristischen Artefakte (Ringing, Treppenstufen, Rauschverstärkung, Überglättung) werden dabei zu einem diagnostischen Vokabular — und genau das braucht man in der Praxis.

---

### 6.5 „Was ist hier kaputt?" — Fehlerdiagnose als Format

Ein Format, das ich für Ihre Vorlesung besonders passend halte, weil es die reale Arbeitserfahrung in dem Feld abbildet.

Sie zeigen ein **fehlerhaftes Ergebnis** und die Studierenden nennen die Ursache. Kandidaten aus Ihrem eigenen Material:

| Fehlerbild | Ursache | Kapitel |
|---|---|---|
| Spektrum mit hellem Kreuz in der Mitte | `fftshift` vergessen | 2 |
| Rekonstruktion mit Wrap-around-Rändern | nicht-zyklische Faltung, zyklisch invertiert | 6, 12 |
| Refokussiertes Bild mit Geisterbildern | falscher Stretch-Faktor $1/\alpha$ | 3 |
| Lichtfeldbild mit Crosstalk zwischen Mikrolinsen | $d_{ML}/b_{ML} \neq d_L/b_L$ | 3 |
| RL-Ergebnis mit verstärktem Rauschen | zu viele Iterationen | 6 |
| ADMM konvergiert nicht | $\rho$ zu klein | 6 |
| Deflection-Map-Gradient rauscht überall | euklidische Distanz statt CMD | 3 |
| Transposed-Conv-Ausgabe mit Schachbrettmuster | Stride/Kernelgröße nicht teilerfremd | 5 |
| Faltungsergebnis systematisch zu hell | Bias in Conv-Layer nicht bedacht (Ihre eigene `Note`!) | 5 |
| SPAD-Histogramm zu früh verschoben | Pile-up durch Totzeit | 11 |

Diese Liste ist eine Prüfungsvorbereitung für sich — es sind genau die Fragen, die in einer mündlichen Prüfung Tiefe von Auswendiggelerntem trennen. Man kann sie über das Semester verteilen (je eine pro Vorlesung als 3-Minuten-Einstieg zur Wiederholung der Vorwoche) oder gebündelt in der letzten Sitzung.

---

### 6.6 Physische Demonstrationen mit Alltagsgegenständen

Der Bruch zwischen Formel und Hörsaal lässt sich an einigen Stellen mit erstaunlich wenig Material überbrücken. Studierende können das meiste mit dem eigenen Handy mitmachen:

| Demo | Material | Zeigt | Kapitel |
|---|---|---|---|
| **Bokeh selbst machen** | Handy, Alufolie mit ausgestanzter Form vor der Linse | Blendenform erscheint als Bokeh | 2 |
| **Lochkamera** | Handy, Pappe mit Nadelloch | unendliche Schärfentiefe, kaum Licht | 2 |
| **Instant-DiffuserCam** | Handy, Frischhaltefolie/Butterbrotpapier/Klebefilm vor der Linse | Kaustik-Bild, das die Kamera nicht mehr fokussiert bekommt | 7 |
| **Bewegungsunschärfe** | Handy mit manueller Belichtungszeit, geschwenkt | das Rect-Kernel als eigene Erfahrung | 8 |
| **Schlieren im Wohnzimmer** | Handy, weit entfernte Punktlichtquelle, Feuerzeug davor | Luftschlieren als Helligkeitsschwankung | 3 |
| **Polarisation** | zwei Polarisationsfilter (oder zwei Sonnenbrillen), LCD-Bildschirm | LCD-Aufbau nachvollziehen | 7 |
| **Displaypixel unterm Mikroskop** | USB-Mikroskop, Handydisplay | Subpixelstruktur als programmierbare Lichtquelle | 3 |
| **Rolling Shutter** | Handy, rotierender Ventilator | Zeitkodierung im Sensor | 8 (Bonus) |

Besonders wirkungsvoll: **Frischhaltefolie vor der Handykamera** zu Beginn von Kapitel 7. Alle machen es gleichzeitig, alle sehen dasselbe Chaos, und Sie sagen: „Das ist Ihr Sensorbild. In den nächsten 90 Minuten rekonstruieren wir daraus das Bild." Das kostet zwei Minuten und rahmt das ganze Kapitel.

Ein Hardware-Vorschlag mit größerer Wirkung: Falls Sie eine **echte Lichtfeldkamera** (Lytro/Raytrix) oder einen Ihrer IOSB-Prototypen (Lichtfeld-Display auf Xperia-Basis, Schlierendeflektometer) in den Hörsaal bringen können — auch nur zum Herumreichen in der Pause — ist der Effekt auf die Motivation erheblich. Bei Kapitel 3 wäre das der natürliche Termin.

---

### 6.7 Kurze schriftliche Formate

**Minute Paper** (letzte 3 Minuten jeder Vorlesung, anonym):
1. Was war heute das Wichtigste?
2. Was ist noch unklar? („Muddiest Point")

Sie bekommen ein Bild vom tatsächlichen Verständnisstand, und die nächste Vorlesung beginnt mit „Drei von Ihnen hatten dieselbe Frage zu $\rho$ — schauen wir das nochmal an." Das ist eine der wenigen Maßnahmen, die praktisch keinen Vorbereitungsaufwand kosten und trotzdem verlässlich wirken. Digital via Particify, oder auf Karteikarten beim Rausgehen.

**Ein-Satz-Zusammenfassung:**
„Erklären Sie den Wiener-Filter in einem Satz, ohne Formel." Zwei Minuten schreiben, drei Beispiele vorlesen lassen. Bei mündlicher Prüfung besonders relevant: genau diese Fähigkeit wird geprüft.

**Erklär's der Nachbarin (Gruppen von drei):**
Person A erklärt Person B ein Konzept, Person C hört zu und darf zwei Nachfragen stellen. Rollen rotieren. Drei Minuten pro Runde. Der Zuhörer-Rolle kommt besondere Bedeutung zu — Fragen stellen ist schwerer, als es aussieht.

---

### 6.8 Jigsaw für parallele Strukturen

Ihr Material enthält mehrere Stellen mit **drei bis vier parallelen Varianten desselben Prinzips**. Das ist die klassische Jigsaw-Konstellation:

| Stelle | Varianten | Zeitbedarf |
|---|---|---|
| Kap. 3, Lichtfeldaufnahme | Mikrolinsenkamera / Gantry / Kamera-Array | 20 min |
| Kap. 3, Deflektionsmessung | 4f-Lichtfeldkamera / Schlierendeflektometer / Laserscanner | 20 min |
| Kap. 7, Modulatoren | Amplitude / Phase / programmierbar / Beleuchtung | 20 min |
| Kap. 10, CASSI | PMVIS / SD-CASSI / DD-CASSI | 20 min |
| Kap. 6, Priors | Tikhonov / $\ell_1$ / TV aniso / TV iso | 15 min |

**Ablauf (am Beispiel CASSI):** Drei Expertengruppen, jede bekommt eine Architektur und einen Fragebogen (Vorwärtsmodell? Ortsauflösung? Rekonstruktion nötig? Vorteil? Nachteil?). Nach 8 Minuten neue Gruppen bilden, in denen je ein Vertreter jeder Architektur sitzt; dort wird gegenseitig erklärt und eine Vergleichstabelle gefüllt. Sie sammeln die Tabelle an der Tafel ein.

Der Effekt: Die Studierenden erzeugen Ihre Vergleichstabelle selbst, statt sie abzuschreiben. Und jede*r hat einmal *erklärt*, nicht nur zugehört.

Kapitel 10 ist dafür besonders geeignet, weil es kurz und stark strukturiert ist — und weil es am 18.02. gemeinsam mit ToF drankommt und da ohnehin Abwechslung guttut.

---

### 6.9 Studierende an die Tafel / an den Regler

Zwei niedrigschwellige Varianten:

**Zwischenschritt an der Tafel:** Bei einer mehrstufigen Herleitung einen Zwischenschritt von einer Person aus dem Publikum machen lassen. Wichtig: einen Schritt wählen, der *sicher* machbar ist (eine Substitution, ein Ausmultiplizieren), nicht den kniffligsten. Sonst wird das Format zur Bloßstellung und niemand meldet sich wieder.

**Regler in Studierendenhand:** Bei einem Widget das Publikum ansagen lassen, was Sie einstellen sollen. „Sagen Sie mir, welches $\rho$ ich probieren soll." Kostet nichts, erzeugt aber Beteiligung — und gelegentlich probiert jemand etwas, worauf Sie nicht gekommen wären.

---

### 6.10 Über die Vorlesungszeit hinaus

**Widget-Aufgaben zwischen den Terminen.** Zwischen zwei Vorlesungen: „Finden Sie mit Widget 8.1 einen Code mit weniger als 4 Nullstellen und über 55 % Lichteffizienz. Bringen Sie Ihre URL mit." Fünf Minuten Aufwand, aber die nächste Vorlesung startet mit Ergebnissen der Studierenden statt mit Ihren Folien.

**Fragen sammeln.** Sie schreiben bereits: „You can send me questions any time." Ein niedrigschwelligeres Gefäß — ein Padlet, ein Forum, ein anonymes Formular — senkt die Hürde deutlich, weil eine E-Mail an den Dozenten für viele Studierende schwerer wiegt als ein Klick.

**Studentische Widget-Beiträge als Bonus.** Ein Widget aus der Stufe-3-Liste als optionale Zusatzleistung. Die entstehenden Artefakte kommen (mit Namensnennung) ins Skript. Das ist erfahrungsgemäß ein starker Motivator und erweitert Ihre Sammlung ohne Ihre Arbeitszeit.

**Screenshot-Galerie.** Wenn Studierende mit Widgets arbeiten, entstehen interessante Zustände. Ein gemeinsames Board („zeigt her, was ihr gefunden habt") macht das sichtbar und erzeugt Anschlusskommunikation.

---

### 6.11 Ein möglicher 90-Minuten-Rhythmus

Zur Orientierung, wie sich das zusammenfügen könnte:

| Zeit | Element | Sozialform |
|---|---|---|
| 0–5 | „Was ist hier kaputt?" — Wiederholung der Vorwoche | Plenum |
| 5–20 | Inhaltsblock 1 | Vortrag |
| 20–25 | ConcepTest mit Peer Instruction | Einzeln → Paar → Plenum |
| 25–40 | Inhaltsblock 2 mit Herleitung | Vortrag |
| 40–45 | Think–Pair–Share am Herleitungs-Stopp | Paar |
| 45–50 | *Pause* | |
| 50–60 | Widget-Demo mit Predict–Observe–Explain | Plenum + Paar |
| 60–75 | Inhaltsblock 3 | Vortrag |
| 75–85 | Wettbewerb oder Jigsaw oder zweiter ConcepTest | Gruppe |
| 85–90 | Minute Paper | Einzeln |

Das sind ca. 30 aktive von 90 Minuten. Der Stoff, den Sie schaffen, wird dadurch weniger — das ist der ehrliche Preis. Bei einer mündlichen Prüfung ist das aber vermutlich der richtige Tausch: weniger Stoff, den die Studierenden erklären können, schlägt mehr Stoff, den sie gesehen haben.

Wenn der Umfang das Problem ist: Kapitel 6 ist mit vier Terminen der offensichtliche Kandidat für Straffung, weil dort mehrere Herleitungen (isotropes TV im Detail, die vollständige Subdifferential-Rechnung) auch als nachlesbarer Anhang funktionieren, während die *Auswahllogik* zwischen den Verfahren die eigentliche Vorlesungsleistung ist.

---

## 7. Konkrete ConcepTest-Fragen zum Sofort-Einsetzen

Zwölf ausformulierte Fragen, je eine pro Vorlesungstermin. Die Distraktoren sind so gewählt, dass sie typische Fehlvorstellungen abbilden — das ist der Punkt, an dem ConcepTests stehen oder fallen.

---

**T1 · Kapitel 2 — Schärfentiefe**

> Sie halbieren den Blendendurchmesser $d$. Was passiert mit dem Durchmesser des Zerstreuungskreises für einen Objektpunkt, der um eine feste Strecke aus der Fokusebene verschoben ist?

- (A) bleibt gleich, er hängt nur von der Defokussierung ab
- (B) halbiert sich ✓
- (C) vervierfacht sich
- (D) viertelt sich

*Distraktor-Logik:* (A) trennt Defokussierung und Blende nicht; (C)/(D) verwechseln mit der Lichtmenge, die tatsächlich quadratisch geht.

---

**T2 · Kapitel 2 — MTF**

> Ein defokussiertes System hat eine Pillbox-PSF. Was gilt für seine MTF?

- (A) monoton fallend, immer positiv
- (B) konstant 1 — Defokus verschiebt nur Phase
- (C) oszillierend mit echten Nullstellen ✓
- (D) monoton steigend

*Warum das gut funktioniert:* (A) ist die Intuition „Unschärfe = Tiefpass" und für Gauß-PSF sogar richtig. Dass die Pillbox Nullstellen erzeugt — und damit *irreversiblen* Informationsverlust — ist genau der Punkt, an dem Kapitel 6 anschließt.

---

**T3 · Kapitel 3 — Lichtfeldkamera**

> Sie ersetzen das Mikrolinsenarray durch eines mit halb so großen Mikrolinsen (gleiche Sensorgröße, gleiche Pixelzahl). Was passiert?

- (A) Orts- und Winkelauflösung verdoppeln sich
- (B) Ortsauflösung verdoppelt sich, Winkelauflösung halbiert sich ✓
- (C) Ortsauflösung halbiert sich, Winkelauflösung verdoppelt sich
- (D) beide bleiben gleich, nur das Sichtfeld ändert sich

*(A) ist der Wunschgedanke, (C) verwechselt die Richtung. Der Trade-off ist die Kernaussage des Abschnitts.*

---

**T4 · Kapitel 3 — Refokussierung**

> Ein Objektpunkt liegt *näher* an der Kamera als die Fokusebene. Welche Steigung hat die zugehörige Struktur im aufgenommenen Lichtfeld $L_b$?

- (A) positiv ✓
- (B) negativ
- (C) null — die Steigung hängt nur von der lateralen Position ab
- (D) unbestimmt, das hängt von der Blende ab

*Direkt an Widget 3.2 gekoppelt: erst abstimmen, dann den Punkt mit der Maus verschieben.*

---

**T5 · Kapitel 4 — Lichttransport**

> Sie beleuchten mit einem einzelnen Projektorpixel $n$ und nehmen ein Bild auf. Was haben Sie gemessen?

- (A) das Element $\mathbf{T}[n,n]$
- (B) die Zeile $\mathbf{T}[n,\cdot]$
- (C) die Spalte $\mathbf{T}[\cdot,n]$ ✓
- (D) die Spur von $\mathbf{T}$

*Zeile-Spalte-Verwechslung ist der häufigste Fehler im ganzen Kapitel und lässt sich hier billig aufklären. Anschlussfrage: „Und wie kämen Sie an die Zeile?" → Helmholtz-Reziprozität.*

---

**T6 · Kapitel 4 — Probing**

> Bernoulli-Probing zur Verstärkung des direkten Anteils, $p = 0{,}125$, $K = 1000$. Um welchen Faktor wird die Diagonale gegenüber den Nebenelementen verstärkt?

- (A) 8 ✓
- (B) 125
- (C) 1000
- (D) 64

*Die Rechnung $\frac{pK}{p^2K} = \frac{1}{p} = 8$ steht im Skript. Dass $K$ herausfällt, ist die eigentliche Einsicht — (C) ist der attraktivste Distraktor.*

---

**T7 · Kapitel 5 — Netze**

> Sie stapeln zehn lineare Layer ohne Nichtlinearitäten dazwischen. Was kann dieses Netz approximieren?

- (A) jede stetige Funktion — es hat ja viele Parameter
- (B) nur lineare Funktionen ✓
- (C) alle stückweise linearen Funktionen
- (D) Polynome bis Grad 10

*(D) ist der reizvollste Distraktor („zehn Schichten → Grad 10"). Die Verkettung linearer Abbildungen ist linear, Punkt — und genau das begründet, warum Nichtlinearitäten nicht optional sind.*

---

**T8 · Kapitel 6 — Wiener-Filter**

> Für welche Frequenzen verhält sich der Wiener-Filter näherungsweise wie der Inversfilter?

- (A) für alle
- (B) für Frequenzen mit hohem SNR ✓
- (C) für Frequenzen mit niedrigem SNR
- (D) nur bei $f = 0$

*Steht wörtlich im Skript und wird trotzdem regelmäßig vertauscht — genau deshalb eine gute Abstimmungsfrage.*

---

**T9 · Kapitel 6 — Richardson–Lucy**

> Sie lassen den RL-Algorithmus statt 30 nun 3000 Iterationen laufen. Das Ergebnis wird …

- (A) besser, RL konvergiert monoton
- (B) unverändert nach Konvergenz
- (C) schlechter, Rauschen wird verstärkt ✓
- (D) numerisch instabil und divergiert

*(A) und (B) sind beides plausible „Konvergenz"-Intuitionen. Ihre Warnung im Text wird hier zur Falle, in die man einmal hineintappen soll — dann sitzt sie.*

---

**T10 · Kapitel 7 — Lensless**

> Warum funktioniert ein blanker Sensor ohne Modulator nicht als Kamera?

- (A) zu wenig Licht erreicht den Sensor
- (B) alle Szenenpunkte erzeugen nahezu dieselbe Sensorantwort ✓
- (C) Beugung zerstört das Bild
- (D) der Sensor ist nicht empfindlich genug

*(B) formuliert genau das Rangdefizit von $\mathbf{H}$ — Anschlussfrage: „Wie sähe $\mathbf{H}$ dann aus?" Antwort: fast alle Spalten identisch.*

---

**T11 · Kapitel 8 — Coded Exposure**

> Der Raskar-Code hat 26 Einsen bei 52 Chops. Was wäre bei 52 Einsen (konventionelle Belichtung)?

- (A) doppelte Lichtmenge und bessere Rekonstruktion
- (B) doppelte Lichtmenge, aber Rekonstruktion praktisch unmöglich ✓
- (C) identische Ergebnisse, der Code ist irrelevant
- (D) halbe Lichtmenge

*Der Zielkonflikt in einer Frage. Direkt gefolgt vom Wettbewerb W1.*

---

**T12 · Kapitel 12 — End-to-End**

> Warum optimiert man die Zwischenvariable $\tilde{h}$ statt der PSF $h$ direkt?

- (A) $\tilde{h}$ hat weniger Parameter
- (B) um $h$ über eine steile Sigmoid nahezu binär zu erzwingen ✓
- (C) weil $h$ nicht differenzierbar ist
- (D) um die Rechenzeit zu senken

*Bezieht sich direkt auf Ihre bestehende `❓`-Box. Wer (C) wählt, verwechselt die Binärheits-*Anforderung* mit einer Differenzierbarkeits-Eigenschaft — ein aufschlussreiches Missverständnis, über das sich gut reden lässt.*

---

## 8. Verzahnung mit Übung und mündlicher Prüfung

### Widgets als Brücke zur Übung

Ihr Übungsplan bietet für jeden Termin einen natürlichen Andockpunkt:

| Übungstermin | Thema | Passendes Widget als Vorbereitung |
|---|---|---|
| 12.11. | Fourier-Transformationen, Mitsuba | 2.5 FFT-Malkasten, 2.8 Fourier-Reihe, 2.1 PSF↔MTF |
| 26.11. | Lichtfeldrechnungen, Mitsuba für Lichtfelder | 3.1 LF-Explorer, 3.2 $(x,u)$-Diagramm, 3.3 Mikrolinsen-Designer |
| 17.12. | Lichttransportrechnungen | 4.1 Transportmatrix-Sandkasten, 4.2 Probing |
| 21.01. | Inverse Probleme | 6.1 Dekonvolutions-Werkbank, 6.3 Soft-Thresholding |
| 04.02. | Inverse Probleme (Fortsetzung) | 6.5 HQS/ADMM-Schrittdurchlauf |
| 18.02. | Diffuser Cam | 7.1 DiffuserCam-Simulator |

Zwei Nutzungsrichtungen:
- **Vorher:** Das Widget baut Intuition, die Übung formalisiert sie. Besonders bei den Lichtfeldrechnungen dürfte das die Fehlerquote deutlich senken.
- **Nachher:** Studierende prüfen ihre eigene Implementierung gegen das Widget. Bei Übung 6 (ADMM-Update-Regeln für DiffuserCam, die Sie im Skript explizit der Übung zuweisen) ist das besonders wertvoll — man sieht sofort, ob ein Vorzeichen falsch ist.

---

### Vorbereitung auf die mündliche Prüfung

Die Prüfungsform verdient eigene Aufmerksamkeit, weil sie andere Fähigkeiten verlangt als eine Klausur — und diese Fähigkeiten muss man üben, nicht nur den Stoff.

**Was in der mündlichen Prüfung wirklich gefragt wird**, ist in aller Regel nicht „Leiten Sie das Soft-Thresholding her", sondern:
- „Erklären Sie mir, was ein Lichtfeld ist." (in Worten, ohne Formel)
- „Warum brauchen wir überhaupt Regularisierung?" (Motivation statt Mechanik)
- „Wann würden Sie ADMM statt HQS nehmen?" (Auswahl statt Rezept)
- „Was passiert, wenn ich $\rho$ vergrößere?" (Parameterabhängigkeit)
- „Wo ist der Zusammenhang zwischen Kapitel 8 und Kapitel 12?" (Struktur)

Alle fünf Sorten werden von den oben beschriebenen Formaten direkt trainiert: Ein-Satz-Zusammenfassungen für die erste, Think–Pair–Share-Stopps für die zweite, Blindverkostung und Werkbank für die dritte, Predict–Observe–Explain für die vierte, Widget Ü1 für die fünfte.

**Zwei konkrete Vorschläge:**

*Prüfungsfragen von Studierenden.* Am Ende jedes Kapitels: „Formulieren Sie zwei Prüfungsfragen zu diesem Kapitel." Eingesammelt, gefiltert, als Sammlung zurückgegeben. Studierende schreiben erstaunlich gute Fragen, sobald sie sich in die Prüferrolle versetzen — und der Perspektivwechsel selbst ist der Lerneffekt.

*Sprech-Übung mit Widget.* In der letzten Übung: Zweiergruppen, eine Person bekommt ein Widget, die andere spielt Prüfer und fragt „Was passiert, wenn …?". Rollen tauschen. Zehn Minuten, und alle haben einmal unter Zeitdruck über Fachinhalte gesprochen — für viele die erste Gelegenheit dazu vor dem Ernstfall.

---

## 9. Nebenbefunde aus der Materialdurchsicht

Kein Auftrag, aber beim Lesen aufgefallen und vermutlich nützlich:

**Kapitel 9 „Quantitative Phase Imaging" fehlt.** Es steht im Inhaltsverzeichnis der Startseite, hat aber kein PDF im Ordner und keinen Slot im Semesterplan. Falls es entfallen soll: aus dem Inhaltsverzeichnis nehmen. Falls es kommen soll: Fourier-Ptychographie böte ein hervorragendes Widget (Beleuchtungswinkel wählen → Verschiebung im Fourierraum → synthetische Apertur wächst sichtbar an; das ist eine der schönsten Visualisierungen, die die Optik zu bieten hat, und passt perfekt zu Ihrem Kapitel-2-Fundament).

**Offene TBD-Stellen:**
- Kap. 3, „Background oriented schlieren: TBD."
- Kap. 3, „TBD: Image showing deflected ray of sight hitting LF probe."
- Kap. 5, letzte Überschrift „Example: Automatic differentiation for inverse Rendering" — Überschrift ohne Inhalt
- Kap. 6, mehrere `\eqref{...}`, die im PDF als `(???)` erscheinen (eq:map_solution, eq:general_inverse_problem, eq:hqs_1, eq:hqs_2, eq:hqs_tv_1, eq:hqs_z_1/2, eq:hqs:iso:1/2/3/11, eq:hqs:ios:11) — vermutlich ein Label-Problem im Jupyter-Book-Build, das sich mit einer sphinx-proof/amsmath-Konfiguration beheben lässt. Für Studierende, die dem Skript folgen, sind kaputte Querverweise ziemlich störend.
- Kap. 6, `image.png` erscheint als roher Dateiname im Ergebnisteil (fehlende Einbindung), und in derselben Tabelle stehen unaufgelöste `$\pm$` und `$\mathbf{f}^2$`.
- Kap. 12, `image-5.png` / `image-6.png` — dieselbe Sache bei den MTF-Vergleichsbildern.

**Formelsatz-Zerfall im PDF.** In den Kapiteln 5 und 6 zerfallen einige mehrzeilige Matrixausdrücke im PDF-Export in verstreute Zeichen (besonders die Batch-Matrix in Kap. 5 und die TV-Definition in Kap. 6). Das ist für die Studierenden, die das PDF zum Lernen nutzen, erheblich störender als eine fehlende Interaktion — und vermutlich mit weniger Aufwand behebbar. Falls das PDF ein wichtiger Nutzungspfad ist, würde ich das noch vor den Widgets angehen.

**Kapitelumfänge und Zeitbudget.** Textumfang der Kapitel (Zeilen im Textexport): Kap. 6 mit 3200, Kap. 3 mit 1890, Kap. 5 mit 1580, Kap. 2 mit 1510 — gegenüber Kap. 8 mit 196, Kap. 9/CASSI mit 280, Kap. 12 mit 420. Der Semesterplan gibt Kapitel 6 vier Termine, aber Kapitel 8, 10, 11 und 12 teilen sich im Wesentlichen zwei. Das ist plausibel gewichtet — aber es bedeutet, dass die Aktivierungsformate in den kurzen Kapiteln besonders dicht sitzen müssen, weil dort wenig Zeit für Wiederholung bleibt. Der Coded-Exposure-Wettbewerb (W1) ist genau deshalb dort platziert.

**Copyright-Jahr.** Alle Kapitel tragen „© Copyright 2022." bei laufendem Semester 2025/26 — vermutlich ein Konfigurationswert, der einmal aktualisiert werden will.

---

*Erstellt am 25. August 2026 auf Basis der PDF-Exporte aus `vl-ci-pdf`.*
