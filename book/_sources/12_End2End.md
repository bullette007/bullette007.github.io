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
from IPython.display import SVG, display, IFrame, HTML, Javascript
import seaborn as sns
import torch
from scipy import ndimage
from scipy import misc
%matplotlib notebook
book = False
```

```{code-cell} ipython3
:init_cell: true

display(HTML('<img id=\"magic\" src=\"figures/1pix.png\" />'))
```

```{code-cell} ipython3
:init_cell: true

%%js
let image = document.getElementById("magic"); 
let ind = image.src.indexOf("xsrf");  
globalThis.myxsrf = image.src.substring(ind+5);
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

# End-to-End Optimization

+++ {"slideshow": {"slide_type": "subslide"}}

##### Content
* Motivation & principle idea
* Laser protection by defocus on purpose
* Learned optimal patterns for deflectometric surface inspection

+++ {"slideshow": {"slide_type": "slide"}}

## Motivation

+++

Think again about the definition of computational imaging systems we gave in the first chapter.

+++

<img src="figures/1/computational_imaging.svg" style="max-height:40vh">

+++

We elaborated, that the joint optimization of all major components (illumination, image acquisition and image processing algorithms) of an artificial vision system should lead to increased performance or other advantages compared to the classical approach of separately optimizing all those components.

+++

###### Principle idea

+++

To be able to jointly optimize all those components, they have to be modeled in a common (mathematical) framework that also allows for modern optimization methods like gradient descent.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
interact(lambda i: showFig('figures/12/end2end_general_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=6), step=1, value=(max_i if book else min_i)))
```

## Laser protection by defocus on purpose

```{code-cell} ipython3
display(HTML('<img id=\"selaci\" src=\"figures/12/selaci_1.svg\" width=80% />'))
f_dft = lambda i: Javascript(f'let image = document.getElementById(\"selaci\"); image.src = \"http://localhost:8888/files/figures/12/selaci_{i}.svg?_xsrf=\" + globalThis.myxsrf')
interact(lambda i: f_dft(i), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=7), step=1, value=(max_i if book else min_i)))
```

The following figure shows the optical setup of the defocused imaging in detail:
<img src="figures/12/selaci_opt_setup.svg" style="max-height:40vh">

+++

### Image formation model

+++

$\begin{align} 
   \mathbf{g} = \mathbf{Hs+n} \,,
\end{align}$

with observation $\mathbf{g}$, matrix $\mathbf{H}$ implementing the convolution, the sought latent image $\mathbf{s}$ and the noise $\mathbf{n}$.

+++

### Image reconstruction via ADMM

+++

**Maximum a-posteriori approach**

$\begin{align} 
   \hat{\mathbf{s}} = \argmin{ \mathbf{s}} \Vert \mathbf{Hs-g} \Vert^2_2 + \tau \Psi (\mathbf{s}) + \mathrm{pos}(\mathbf{s}) \,,
\end{align}$

+++

with balanced regularization $\tau \Psi( \mathbf{s})$ and enforced positivity $\mathrm{pos}(\mathbf{s})$.

+++

**Introduction of slack variables**

$\begin{align} 
  \hat{\mathbf{s} }, \hat{\mathbf{u} }, \hat{\mathbf{w} } &= \argmin{\mathbf{s}, \mathbf{u}, \mathbf{w} } \, \frac{1}{2} \Vert \mathbf{Hs-g} \Vert^2_2 + \tau \Psi( \mathbf{u} ) + \mathrm{pos}( \mathbf{w}) \\
  & \mathrm{s.t.}\quad \mathbf{s-u = 0}, \quad \mathbf{s-w = 0} \,,
\end{align}$

with slack variables $\mathbf{u,w}$.

+++

**Transformation into augmented Lagrangian in scaled form**

$\begin{align} 
  L_{\tau, \alpha, \beta}&( \mathbf{s,u,w, \tilde{u}, \tilde{w}}) = \frac{1}{2} \Vert \mathbf{Hs-g} \Vert^2_2  \\
  &+ \tau \Psi(\mathbf{u}) + \frac{\alpha}{2} \Vert \mathbf{s-u+\tilde{u}} \Vert^2_2 + \frac{\alpha}{2} \Vert \mathbf{\tilde{u}} \Vert^2_2 \\
  &+ \mathrm{pos}( \mathbf{w}) + \frac{\beta}{2}\Vert \mathbf{s-w+\tilde{w}} \Vert^2_2 + \frac{\beta}{2} \Vert \mathbf{\tilde{w}} \Vert^2_2 \,,
\end{align}$
with the dual variables $\mathbf{\tilde{u}, \tilde{w}} $ of $\mathbf{u}, \mathbf{w} $ and the corresponding weights $\alpha, \beta$.

+++

**Anisotropic total variation regularization**

\begin{align}
  \Psi( \mathbf{s}) = \Vert \mathbf{Ds} \Vert_1 \,,
\end{align}
with finite difference operator $ \mathbf{D} = \left[\mathbf{D}\transp_x \mathbf{D}\transp_y \right]\transp \in \mathbb{R}^{2N \times N}$.

+++

**Update rules**

\begin{align}
  \mathbf{s}_{i+1} \leftarrow &\left( \mathbf{H}\transp \mathbf{H} + \alpha \mathbf{D}\transp \mathbf{D} + \beta \mathbf{I} \right)^{-1} \\ &\cdot \left( \mathbf{H}\transp \mathbf{g} - \alpha \mathbf{D}\transp (\mathbf{\tilde{u}}_{i} - \mathbf{u}_i ) - \beta ( \mathbf{\tilde{w}}_i - \mathbf{w} ) \right) \,, \\
  \mathbf{u}_{i+1} \leftarrow &\,\mathcal{S}_{\tau / \alpha}( \mathbf{Ds}_{i+1} + \mathbf{\tilde{u}}_i ) \,, \\
  \mathbf{w}_{i+1} \leftarrow &\mathrm{max}( \mathbf{0}, \mathbf{s}_{i+1} + \mathbf{\tilde{w}}_i ) \,, \\
  \mathbf{\tilde{u}}_{i+1} \leftarrow & \mathbf{\tilde{u}}_i + \mathbf{Ds}_{i+1} - \mathbf{u}_{i+1} \,, \\
  \mathbf{\tilde{w}}_{i+1} \leftarrow & \mathbf{\tilde{w}}_i + \mathbf{s}_{i+1} - \mathbf{w}_{i+1} \,,
\end{align}
with $i\in \mathbb{N}$ indexing the iterations, $ \mathbf{I}$ denoting the identity matrix and the element-wise soft thresholding operator $\mathcal{S}_{\tau / \alpha}$.

+++

**Problem: cyclic convolution assumption**

Deconvolution step in Fourier domain $\rightarrow$ cyclic convolution implicitly assumed.<br>
⚡<br>
Physical / optical convolution due to defocus is of course not cyclic!

+++

❓ **Question**

Assume that you only have access to a spatial convolution operator that does not support the "wrap around" border handling. How could you still make sure that the result is equal to a cyclic convolution?

+++

### Optical zero padding

+++

<img src="figures/12/optZeroPadding.svg" style="max-height:40vh">
with sensor size $d_\mathrm{s}$, defocus distance $\Delta$, stop diameter $d_\mathrm{f}$, focal length $f$ and lens diameter $d_\mathrm{L}$.

+++

The diameter $d_\mathrm{f}$ of the required step is given by
\begin{align}
  d_ \mathrm{f} = 2 \left( \frac{\frac{d_ \mathrm{L} + d_ \mathrm{s}}{2}}{f + \Delta} \cdot f - \frac{d_ \mathrm{L}}{2} \right) \,.
\end{align}

+++

### Optimization pipeline

```{code-cell} ipython3
display(HTML('<img id=\"selaci_optPipe\" src=\"figures/12/selaci_optPipeline_1.svg\" width=80% />'))
f_optPipe = lambda i: Javascript(f'let image = document.getElementById(\"selaci_optPipe\"); image.src = \"http://localhost:8888/files/figures/12/selaci_optPipeline_{i}.svg?_xsrf=\" + globalThis.myxsrf')
interact(lambda i: f_optPipe(i), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=6), step=1, value=(max_i if book else min_i)))
```

+++

### Optimization of the PSF

+++

**Constraint for light efficiency:**

+++

❓ **Question**

Do you have an idea how a loss function for the PSF could look like that is minimized when a certain user-definable fraction $p$ of light is transmitted by the amplitude mask?

+++

$\begin{align} 
  \ell_{\mathrm{PSF} } (h) =  \left( \frac{\sum_{\mathbf{x} \in \Omega} h(\mathbf{x} )}{\vert \Omega \vert} - p \right)^2 \,,
\end{align}$

with the discrete spatial support $\Omega$ of the PSF and the sought light efficiency ratio $p\in \left(0,\ldots, 1 \right)$.

+++

This loss term reaches its minimum when the mean of the coefficients of the PSF approaches $p$, i.e., when a ratio $p$ of the incident light is transmitted.

+++

**Binary coefficients**

+++

❓ **Question**

What could be done to force the values of the PSF to be close to either $0$ or $1$? Think about a function you got to know that processes its input in that manner ...

+++

Optimize intermediate variable $\tilde{h}$ instead of $h$ directly.

+++

Feed $\tilde{h}$ into steep sigmoid function:

\begin{align}\label{eq:psf_update}
  h(\mathbf{x} ) \leftarrow \frac{1}{1+\exp(-\gamma \tilde{h}( \mathbf{x} ))} \,,
\end{align}
with a scaling factor $\gamma$ of, e.g., $\gamma = 20$. <br>
Then $h$ will be approximately binarized.