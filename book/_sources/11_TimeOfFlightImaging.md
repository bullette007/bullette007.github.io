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
from scipy import ndimage
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

# Time-Of-Flight Imaging

+++ {"slideshow": {"slide_type": "subslide"}}

##### Content
* Introduction
* Transient imaging
* Non-line-of-sight imaging
* TBC.

+++ {"slideshow": {"slide_type": "slide"}}

## Introduction

+++

Time-of-flight imaging techniques try to capture light while it is in motion or they try to measure the light's travel time inside a scene with extreme accuracy. 

The resulting information can be exploited to enable several novel applications in various fields, e.g., in robotic vision, remote sensing, medical imaging, autonomous driving, etc.

+++

Key enablers for these approaches are ultrashort pulse lasers and high-sensitive light detectors.

+++

### Ultrashort pulse lasers

+++

**Ultrashort pulse lasers** (UPL) can emit light pulses of very short duration, typically of the order of femtoseconds ($10^{-15}$s) to one picosecond ($10^{-12}$).

Common ultrashort pulse lasers are based on Ti:sapphire crystals or dyes.

+++

### Single-photon avalanche diode

+++

So-called **single-photon avalanche diodes** (SPADs) are semiconductors that are similar to common photodiodes.

In a photodiode, a low bias voltage is used so that, due to the photoelectric effect, arriving photons cause a leakage current that increases linearly with the number of arriving photons. The linearity is exploited to perform quantitative measurements of the incident light's intensity.

+++

In a SPAD, the bias voltage is set so high, that even a single arriving photon can cause an avalanche of electrons to be released from the surrounding bulk material leading to a corresponding current.

The auxiliary electronics working along with a SPAD have to correctly sense the increasing current, generate a synchronous output signal, lower the bias voltage to quench the avalanche and restore the initial operating conditions.

As this takes some time, the SPAD is not sensitive for further photons for a so-called *dead time* of tens to hundreds of nanoseconds.

+++

When properly synchronized, ultrashort pulse lasers and SPADs can be employed to precisely measure the time light travels inside the observed scene. Via the speed of light $c$, this time can be converted into a geometric distance.

+++

This is also the principle used in so-called **light detection and ranging** (lidar) devices to capture a point cloud of the observed scene containing the distances to the measured points.

+++

The typical output value of a SPAD synchronized with an UPL is a time value corresponding to the duration between the emission of the laser pulse and the detection of a photon by the SPAD.

+++

As SPADs usually only measure the travel time of one photon for one emitted laser pulse due to the comparatively long dead time, usually millions of laser pulses are emitted (at MHz rates) and the corresponding measured travel times are collected in a histogram $h$.

+++

When emitting a laser pulse into the scene, a highly precise timer is started that is stopped as soon as the SPAD registers an event. The measured time is then converted into a digital number by a so-called *time-to-digital converter* (TDC).

+++

SPADs are commercially available as:
* Single pixel sensors <br> A full image is obtained via a two-dimensional scanning of the scene or by optical coding.
* One-dimensional arrays <br> Only a one-dimensional scanning is required scan a volume of the scene.
* Two-dimensional arrays <br> No additional scanning is required. Unfortunately, the large footprint of currently available SPADs severely limits the spatial resolution when used in a two-dimensional array without optical scanning.

+++

## Transient imaging

+++

So-called *transient images* are images of a scene captured at certain points in time while a pulse of light is still traveling through the scene. Typical cameras integrate over all transient images that are created by a scene due to their comparatively long exposure time when compared to the speed of light.

+++

Transient imaging techniques employ SPADs and UPLs in concert to reconstruct transient images at different time steps. Transient images can reveal interesting properties (i.e., different events of scattering and light redirections) of the scene.

+++

The values $\tau$ of a pixel of a set of transient images corresponding to some time duration can be imagined as a time impulse response function, i.e., the temporal intensity response of the scene to a pulse of light.

```{code-cell} ipython3
interact(lambda i: showFig('figures/11/transient_imaging_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=0),max=(max_i:=5), step=1, value=(max_i if book else min_i)))
```

### Forward model of transient imaging

+++

After emitting a laser pulse into the scene, the amount of light scattered back to the detector is a temporally varying distribution of photons $g$. The photon flux $r$ incident on the detector during time interval $t$ is given by:

$\begin{align} 
   r(t) = (\tau * g)(t) + a(t) \,,
\end{align}$

with $\tau$ denoting the temporal impulse response of the scene and with the ambient photon flux $a(t)$.

+++

The temporal impulse response $\tau$ incorporates all optical effects of the scene that influence the travel paths / time of the laser pulse (e.g., reflectance, scattering, etc.).

+++

Assume a scene where light would bounce only once before reaching the detector, i.e., with only direct light transport. In this case, $\tau$ would be a Dirac delta function.

+++

Conversely, for global light transport (i.e., with caustics, complex scattering events, interreflections etc.), $\tau$ models the corresponding temporal impulse response of the scene.

+++

The ideal photon counter would sample the rate function 

$\begin{align} 
   \lambda (t) = \eta \left( r * f \right) (t) + d \,,
\end{align}$

with $\eta \in [0,1]$ representing the sensor's quantum efficiency and the avalanche probability of the SPAD, $d$ denoting the dark count rate (number of false detections) in Hz and $f$ being the temporal jitter (about tens or a few hundreds of ps for state-of-the-art SPADs).

+++

An event registered by the SPAD does not necessarily have to be the first arriving photon. Whether a photon is detected within a short time window is a Bernoulli trail with the two possible outcomes of photon detected and no photon detected.

+++

By repeating this Bernoulli trial for $N$ times by emitting $N$ laser pulses, the histogram $h$ of the photon travel times is built up.

The probability of detecting $h(t)$ photons can be modeled as a Poisson distribution:

$\begin{align} 
   h(t) \sim \mathcal{P} (N \lambda(t))\,,
\end{align}$

with the expected number of photons $\lambda(t)$ at time $t$.

+++

### Reconstruction of transient images

+++

The reconstruction of transient images from the measured noisy and blurry histograms can be modeled as a deconvolution problem in the presence of Poisson noise.

For this means, we vectorize the involved quantities as follows:
* the temporal impulse response $\boldsymbol{\tau} \in \mathbb{R}^{n_x n_y n_t}$, i.e., the sought latent transient image, 
* the measured histogram $\mathbf{h}\in \mathbb{R}^{n_x n_y n_t}$ and
* the dark count $\mathbf{d} \in \mathbb{R}^{n_x n_y n_t}$.

+++

With the measurement matrix $\mathbf{A} \in \mathbb{R}^{n_x n_y n_t \times n_x n_y n_t}$ encoding the convolution of the transient image with the laser pulse $g$ and the SPAD jitter $f$, we can express

$\begin{align} 
   h \sim \mathcal{P}(\mathbf{A} \boldsymbol{\tau} + \mathbf{d}) \,.
\end{align}$

The transient images have spatial resolution of $n_x \times n_y$ and for each pixel there are $n_t$ time bins in the histogram.

+++

The reconstruction problem can be formulated as a maximum likelihood estimation with the constraint of non-negative solutions:

$\begin{align} 
   \boldsymbol{\tau}^* = \argmin{\boldsymbol{\tau}} -\log \left( p(\mathbf{h} \vert \mathbf{A} \boldsymbol{\tau}) \right) + \Psi (\boldsymbol{\tau}) \,, \\
   \text{subject to } \, \boldsymbol{\tau} \geq \mathbf{0} \,,
\end{align}$

with the likelihood $p(\mathbf{h} \vert \mathbf{A} \boldsymbol{\tau})$ of measuring the histogram $\mathbf{h}$ for a given transient image $\boldsymbol{\tau}$ and $ \Psi (\boldsymbol{\tau})$ representing a suitable regularizer.

+++

In order to solve this optimization problem with ADMM, first the optimization objectives are split into independent terms via slack variables $\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3 \in \mathbb{R} ^{n_x n_y n_t}$ and the corresponding constraints are added:

$\begin{align} 
  \boldsymbol{\tau}^* = \argmin{\boldsymbol{\tau}, \mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3} -\log \left( p(\mathbf{h} \vert \mathbf{z}_1) \right) + \mathrm{pos}(\mathbf{z}_2) + \Psi (\mathbf{z}_3) \,, \\
  \text{subject to } \, \mathbf{A}\boldsymbol{\tau} = \mathbf{z}_1, \boldsymbol{\tau} = \mathbf{z}_2, \boldsymbol{\tau} = \mathbf{z}_3 \,,
\end{align}$

with 

$\begin{align} 
  \mathrm{pos}(\mathbf{x})=
  \begin{cases}    
    +\infty \quad &\text{if } x_i < 0 \text{ for any } i\\
    0 &\text{otherwise.}
  \end{cases}   
\end{align}$

+++

Then, the augmented Lagrangian in scaled form of the objective can be expressed with the Lagrange multipliers $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ corresponding to the three constraints and with the corresponding scalar weights $\mu_1, \mu_2, \mu_3$:

$\begin{align} 
   L(\boldsymbol{\tau}, \mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3, \mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3) = &-\log \left( p(\mathbf{h} \vert \mathbf{z}_1) \right) + \mathrm{pos}(\mathbf{z}_2) + \Psi (\mathbf{z}_3) \\
   & + \frac{\mu_1}{2} \left\|\mathbf{A}\boldsymbol{\tau} - \mathbf{z}_1 + \mathbf{u}_1  \right\| ^2_2 - \frac{\mu_1}{2} \left\| \mathbf{u}_1 \right\| ^2_2  \\
   & + \frac{\mu _2}{2} \left\| \boldsymbol{\tau} - \mathbf{z}_2 + \mathbf{u}_2 \right\| ^2_2 - \frac{\mu _2}{2} \left\| \mathbf{u}_2 \right\| ^2_2 \\
   & + \frac{\mu _3}{2} \left\| \boldsymbol{\tau} - \mathbf{z}_3 + \mathbf{u}_3 \right\| ^2_2 - \frac{\mu _3}{2} \left\| \mathbf{u}_3 \right\| ^2_2 \\
\end{align}$

+++

### Results

+++

Experimental results achieved with the described approach are reported in the corresponding paper [Reconstructing Transient Images from Single-Photon Sensors](https://www.computationalimaging.org/publications/reconstructing-transient-images-from-single-photon-sensors-cvpr-2017/) by Matthew O'Toole et al.

+++

## Non-line-of-sight imaging

+++

As mentioned in the introduction, many applications like robotics, medicine, autonomous driving, etc. could greatly benefit from knowing in advance what is behind a corner or other structure blocking the direct line of sight.

Non-line-of-sight imaging methods can provide that information by emitting ultra short laser pulses into the scene and by measuring the time elapsed until corresponding photons return from the scene.

```{code-cell} ipython3
interact(lambda i: showFig('figures/11/nlos_task_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=0),max=(max_i:=6), step=1, value=(max_i if book else min_i)))
```

```{code-cell} ipython3

```
