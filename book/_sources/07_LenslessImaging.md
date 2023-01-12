---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: compimg
  language: python
  name: compimg
---

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

%%js
var cells = Jupyter.notebook.get_cells();
           for (var i = 0; i < cells.length; i++) {
               var cur_cell = cells[i];
               var tags = cur_cell._metadata.tags;
               console.log(i);
               console.log(tags == undefined);
               console.log(tags);
               if (tags != undefined) {
               for (var j = 0; j < tags.length; j++) {
                  if (tags[j]=="book_only" | tags[j]=="remove-cell") {cur_cell.element.hide();}}
               for (var j = 0; j < tags.length; j++) {
                   if (tags[j]=="presentation_only") {cur_cell.element.show();}}
            }}
```

```{code-cell} ipython3
---
init_cell: true
slideshow:
  slide_type: notes
tags: [remove-cell]
---
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as sig
from scipy.integrate import simpson
from IPython.display import SVG, display, IFrame, HTML
import seaborn as sns
import torch
from scipy import misc
%matplotlib notebook
book = False
```

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
book = True
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

%%javascript
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
});
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

%%javascript

MathJax.Hub.Queue(
  ["resetEquationNumbers", MathJax.InputJax.TeX],
  ["PreProcess", MathJax.Hub],
  ["Reprocess", MathJax.Hub]
);
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

def showFig(path,i,ending, width, height):
    filename = path+str(i)+ending
    return HTML("<img src=\"" + filename +  f"\" style=\"max-height:{height}vh\"/>")
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

def showFig2(path,i,ending, width, height):
    imgToShow = plt.imread(f"{path}{i}{ending}")
    plt.imshow(imgToShow)
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

def imshow(img, cmap=None):
    plt.close('all')
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()
```

```{code-cell} ipython3
:init_cell: true
:tags: [remove-cell]

def imshow2(img, cmap=None):
    #plt.close('all')
    #plt.figure()
    plt.clf()
    plt.imshow(img, cmap=cmap)
    #plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

$\begin{align}
  \newcommand{transp}{^\intercal}
  \newcommand{F}{\mathcal{F}}
  \newcommand{Fi}{\mathcal{F}^{-1}}
  \newcommand{inv}{^{-1}}
  \newcommand{stochvec}[1]{\mathbf{\tilde{#1}}}
  \newcommand{argmax}[1]{\underset{#1}{\mathrm{arg\, max}}\,}
  \newcommand{argmin}[1]{\underset{#1}{\mathrm{arg\, min}}\,}
\end{align}$

<font size="7"> Computational Imaging </font><br><br><br>

+++ {"slideshow": {"slide_type": "-"}}

# Lensless Imaging

+++ {"slideshow": {"slide_type": "subslide"}}

##### Content
* TBD

+++ {"slideshow": {"slide_type": "slide"}}

## Introduction into lensless imaging

+++

As the name suggests, lensless imaging setups capture light on a sensor without focusing it using a lens (system). 

* As a direct consequence, the image formed on the sensor is usually not interpretable for the naked human eye. 
* Only with the help of a suitable reconstruction algorithm, the sought image of the observed scene can be obtained.

+++

Lensless imaging systems have some advantages compared to their lens-based counterparts:

* Reduced weight and size,
* reduced cost,
* scalability,
* field of view,
* visual privacy,
* potential applicability to ranges of non-visible wavelengths where it is difficult or impossible to build lenses, e.g., x-rays (provided a suitable sensor).

+++

Disadvantages of lensless imaging systems:

* Image quality,
* need for computational image reconstruction (also results in increased power consumption, longer processing times, etc.),
* reduced amount of collected light.

+++

## Modulating light in a lensless imaging system

+++

Instead of using lenses, lensless imaging systems use other techniques to modulate the light so that each point of the observed scene yields a different system response. This is a necessary condition for the reconstruction methods to be able to recover the sought sharp image of the scene. 

+++

The optical effect of relevant modulators is usually modeled via the corresponding point spread function (PSF).

+++

Just using a sensor without any modulator results in a severely ill-posed problem that can barely be solved, as every scene point approximately leads to the same system response.

+++ 

Typical modulators can be categorized into the following groups:

* Amplitude mask modulators (block, attenuate or transmit light with a spatially varying pattern),
* phase mask modulators (change the phase of transmitted light with spatially varying phase shifts),
* programmable modulators (change amplitude or phase of transmitted light with a spatially varying pattern that can be computationally controlled) and
* illumination modulators (controlling, patterning the illumination).

+++

### Amplitude modulators

+++

<img src="figures/7/amplitude_modulators.svg" style="max-height:40vh">

+++

Amplitude modulators are usually produced by photo-lithographically etching a binary pattern of reflective chrome on a glass substrate or by printing a binary pattern with dark ink on a thin transparent film.

+++

The regions covered with chrome or ink reflect or absorb incident light (i.e., they block it) whereas light is transmitted by the other regions.

+++

Amplitude modulators have their roots in x-ray imaging where it is difficult to construct lenses.

+++

Light modulation by amplitude masks is modeled in two ways, depending on the distance $d$ between the mask and the sensor: 

+++

1. Small $d$: The point spread function (PSF) is modeled based on the shadow of the mask cast on the sensor.

+++

2. Large $d$: The PSF is modeled based on the diffraction of incident light at the mask.

+++

The so-called *Fresnel number* $N_\mathrm{F}$ helps with distinguishing between the two cases and is defined as:

$\begin{align} 
   N_\mathrm{F} = \frac{a^2}{d\lambda} \,,
\end{align}$

with $a$ denoting the size of the mask's smallest open region and $\lambda$ denoting the shortest wavelength of the involved light.

+++

For $N_\mathrm{F} >> 1$, the assumptions of geometric optics (i.e., ray optics) hold and the first model can be applied. If $N_\mathrm{F} <= 1$, diffraction has to be taken into account.

+++

The major disadvantage of amplitude masks is the reduced light efficiency caused by blocking parts of the incident light. This can lead to poor signal-to-noise ratios, especially in low-light applications.

+++

### Phase modulators

+++

Phase modulators change the relative optical path length between different light rays reaching the mask. This introduces changes in the direction of propagation of transmitted light rays, so that distributions of spatially varying light intensities arise behind the mask (so-called *caustics*). 

+++

Hence, phase modulators can yield quite sophisticated intensity patterns varying with respect to the incident light and therefore represent light modulators suitable for lensless imaging.

Furthermore, they do not block incident light what leads to high signal-to-noise ratios.

+++

There are three different main kinds of phase modulators: phase gratings, diffusers and phase masks.

Phase gratings and phase masks and can be produced by photo-lithographically etching structures into glass or by an additively polymerizing photoresist on a transparent substrate. 