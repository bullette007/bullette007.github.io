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
                  if (tags[j]=="book_only" | tags[j]=="remove-cell") {cur_cell.element.hide();
                  if (tags[j]=="presentation_only") {cur_cell.element.show();}
            }}}
```

```{code-cell} ipython3
%%js
Jupyter.notebook.get_cell(0).element.hide()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
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
%matplotlib notebook
```

LaTeX-macros:
$\begin{align}
  \newcommand{transp}{^\intercal}
  \newcommand{F}{\mathcal{F}}
  \newcommand{Fi}{\mathcal{F}^{-1}}
  \newcommand{inv}{^{-1}}
  \newcommand{stochvec}[1]{\mathbf{\tilde{#1}}}
  \newcommand{argmax}{\mathrm{arg\, max}}
\end{align}$

+++

LaTeX command stochvec:
$\newcommand{stochvec}[1]{\mathbf{\tilde{#1}}}$
<br>
LaTeX command transp:
$\newcommand{transp}{^\intercal}$
LaTeX command argmax:
$\newcommand{argmax}[1]{\underset{#1}{\mathrm{arg\, max}}}$
LaTeX command argmin:
$\newcommand{argmin}[1]{\underset{#1}{\mathrm{arg\, min}}}$

```{code-cell} ipython3
%%javascript
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
});
```

```{code-cell} ipython3
%%javascript

MathJax.Hub.Queue(
  ["resetEquationNumbers", MathJax.InputJax.TeX],
  ["PreProcess", MathJax.Hub],
  ["Reprocess", MathJax.Hub]
);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def showFig(path,i,ending, width, height):
    return IFrame("./" + path+str(i)+ending, width, height)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def showFig2(path,i,ending, width, height):
    imgToShow = plt.imread(f"{path}{i}{ending}")
    plt.imshow(imgToShow)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def showFig(path,i,ending, width, height):
    #return SVG(filename = "./" + path+str(i)+ending)
    display(SVG(filename = "./" + path+str(i)+ending))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def showFig(path,i,ending, width, height):
    filename = path+str(i)+ending
    return HTML("<img src=\"" + filename +  "\"/>")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def imshow(img, cmap=None):
    plt.close('all')
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: notes
---
def imshow2(img, cmap=None):
    #plt.close('all')
    #plt.figure()
    plt.clf()
    plt.imshow(img, cmap=cmap)
    #plt.show()
```

```{code-cell} ipython3
#interact(lambda i: showFig('figures/2/dftSpectrum_',i,'.svg',800,700), i=widgets.IntSlider(min=1,max=3, step=1, value=1))
```

```{code-cell} ipython3
#<img src="figures/2/pinholePrinciple.svg" style="width:40vw">
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
interact(lambda i: showFig2('figures/3/lfSAIs/out_09_',i,'.png',800,1100), i=widgets.IntSlider(min=0,max=16, step=1, value=1))
```

+++ {"slideshow": {"slide_type": "fragment"}}

<img src="figures/3/SchlierenDeflectometer.svg" style="max-height:40vh">

+++ {"slideshow": {"slide_type": "notes"}}

# Inverse Problems in Computational Imaging

+++ {"slideshow": {"slide_type": "slide"}}

<font size="7"> Computational Imaging </font><br><br><br>
<font size="5"> Chapter 5: Inverse Problems in Computational Imaging </font>

+++ {"slideshow": {"slide_type": "subslide"}, "tags": ["book_only"]}

##### Content
* Introduction into inverse problems
    * What are inverse problems?
    * Approaches for solving inverse problems
* Image deconvolution
    * Inverse filter
    * Wiener filter (classical) and its problems
    * Richardson-Lucy
    * Optimization-based
        * Half-quadratic Splitting
    * Deep SNR-estimation for Wiener filter
    * ADMM

+++

## Introduction into inverse problems

+++

## Image deconvolution

+++

Image deconvolution deals with the problem of recovering an original image $x$ after it has been convolved with a kernel $c$ and has been affected by some additive noise $n$.<br> Hence, the image formation model can be expressed via:
$\begin{align}\label{eq:img_formation}
    b=c*x + n\,,
\end{align}$
with $b$ denoting the observed image.

<br> Note: The dependence on a spatial variable (e.g., like $\mathbf{x}$ for $g(\mathbf{x})$) is omitted for this chapter for the sake of clarity.

+++

The task is to find an $\hat{x}$ that is as close to the original $x$ as possible.

+++

Methods and approaches dealing with deconvolution can be divided into two groups:
1. *Blind* deconvolution: The kernel $c$ is assumed to be unknown and also has to be recovered / estimated.
2. *Non-blind* deconvolution: The kernel $c$ is assumed to be known.

<br>Note that there is some research which focuses on estimating the kernel $c$ and then employs non-blind deconvolution methods to perform the deconvolution.

+++

According to the convolution theorem of the Fourier transform, equation $\eqref{eq:img_formation}$ can be expressed as:
$\begin{align}\label{eq:img_formation_fourier}
    b = \mathcal{F}^{-1}\lbrace \mathcal{F} \lbrace c \rbrace \cdot \mathcal{F} \lbrace x \rbrace \rbrace + n
\end{align}$

+++

**Important note:** Equations $\eqref{eq:img_formation}$ and $\eqref{eq:img_formation_fourier}$ only result in the same numbers, if the convolution is performed with ciruclar boundary conditions, i.e., when the convolution kernel wraps around the image borders.

+++

##### Duality between signal processing perspective and algebraic perspective

+++

Since the convolution is a linear operator, it can also be expressed in terms of a matrix vector multiplication:
$\begin{align}\label{eq:duality}
    b = c*x \Leftrightarrow \mathbf{b} = \mathbf{C}\mathbf{x}\,,
\end{align}$
with a square circulant Toeplitz matrix $\mathbf{C}\in \mathbb{R}^{N\times N}$ implementing the convolution and $\mathbf{x,b} \in \mathbb{R}^N$ representing the vectorized forms of the corresponding images $x$ and $b$.

+++

### Inverse filter

+++

The inverse filter represents the straightforward approach of solving equation $\eqref{eq:img_formation_fourier}$ for $x$ while neglecting the noise $n$:
$\begin{align}\label{eq:inv_filter}
        \hat{x}_{\mathrm{inv}}=\Fi\left\lbrace \frac{\F\lbrace b \rbrace}{\F\lbrace c \rbrace } \right\rbrace \,.
\end{align}$

+++

Properties of the inverse filter:
* computationally efficient,
* problematic for small values of $\F \lbrace c \rbrace$, (amplifies noise).

Unfortunately, the second point is true for most practically relevant point spread functions.

+++

TODO: Example images.

+++

EXERCISE: Implement inverse filter and do some tests for varying noise levels.

+++

### Wiener filter

+++

TBD

+++

### Richardson-Lucy

+++

TBD

+++

### Optimization-based methods

+++

As stated before, deconvolution is an ill-posed problems, i.e., usually there are infinitely many solutions which satisfy the measurements.

$\Rightarrow$ One needs an instrument to determine how desirable a given solution is.

+++

#### Bayesian perspective of inverse problems

+++

General formulation of an inverse problem:
$\begin{align}
    \mathbf{b}=\mathbf{A}\mathbf{x} + \mathbf{n}\,,
\end{align}$
with $\mathbf{x}\in \mathbb{R}^N, \mathbf{b,n}\in \mathbb{R}^M, \mathbf{A}\in \mathbb{R}^{M\times N}$.

+++

Interpret as random vectors:
* $\stochvec{x}\sim \mathcal{N}(\mathbf{x},0)$,
* $\stochvec{n}\sim \mathcal{N}(0,\sigma^2)$,
* $ \stochvec{b}\sim \mathcal{N}((\mathbf{Ax}),\sigma^2)$

+++

For probability of observation $ \mathbf{b}$ it holds:
$\begin{align}
    p( \mathbf{b}\vert \mathbf{x}, \sigma) \propto \exp \left( - \frac{\Vert \mathbf{b} - \mathbf{Ax} \Vert^2_2}{2\sigma^2} \right)
\end{align}
$

+++

According to Bayes' rule:
$\begin{align}
    \underbrace{p(\mathbf{x}\vert \mathbf{b},\sigma)}_{\mathrm{posterior}} = \frac{p( \mathbf{b}\vert \mathbf{x}, \sigma)\cdot p( \mathbf{x})}{p( \mathbf{b})} \propto \underbrace{p( \mathbf{b} \vert \mathbf{x}, \sigma)}_{\mathrm{image\ formation\ model}} \cdot \underbrace{p( \mathbf{x})}_{prior}\,.
\end{align}
$

+++

Maximum-a-posteriori (MAP) solution:
$\begin{align}
  \hat{x}_\mathrm{MAP} &= \argmax{\mathbf{x}}\, p(\mathbf{x}\vert \mathbf{b},\sigma) \\
                         &= \argmax{\mathbf{x}}\, \log p(\mathbf{x}\vert \mathbf{b},\sigma) \\
                         &= \argmin{ \mathbf{x}}\, - \log p(\mathbf{x}\vert \mathbf{b},\sigma) \\
                         &= \argmin{ \mathbf{x}}\, - \log p( \mathbf{b} \vert \mathbf{x}, \sigma) - \log p( \mathbf{x}) \\
                         \label{eq:map_solution}&= \argmin{ \mathbf{x}}\, \underbrace{\frac{\Vert \mathbf{b} - \mathbf{Ax} \Vert^2_2}{2\sigma^2}}_{\mathrm{data\, fidelity\, term}} + \underbrace{\Psi(\mathbf{x})}_{ \mathrm{regularizer}}\,.                         
\end{align}
$

+++

The choice of image priors / regularizers depends on the imaging task, i.e., the nature of the images that are to be recovered. Examples are:
* Blurry imges $\rightarrow$ promote smoothness $\Psi (\mathbf{x}) = \left\| \underbrace{\Delta}_{\text{Laplace operator}} \mathbf{x} \right\|_2 $
* Sparse images (e.g., stars) $\rightarrow$ promote sparsity $\Psi (\mathbf{x}) = \left\| \mathbf{x} \right\|_1$
* Natural images $\rightarrow$ promote spares gradients $\Psi \mathrm{TV}(\mathbf{x})$

+++

##### Total Variation

The intuition behind total variation is that in natural images, regions of almost constant intensities are separated by sharp edges. Hence, the gradient of such images can be assumed to be sparse.

The gradient is calculated by means of convolutions with the finite difference operators in $x$- and $y$-direction:

* Finite difference in $x$-direction: $d_x * x = \begin{pmatrix} 0 & 0 & 0 \\ 0 & -1 & 1 \\ 0 & 0 & 0  \end{pmatrix} * x = \mathbf{D}_x \mathbf{x}$
* Finite difference in $y$-direction: $d_y * x = \begin{pmatrix} 0 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 1 & 0  \end{pmatrix} * x = \mathbf{D}_y \mathbf{x}$

+++

* Anisotropic: 
$\begin{align} 
  \mathrm{TV}(\mathbf{x}) &= \left\| \mathbf{D}_x \mathbf{x} \right\|_1 + \left\| \mathbf{D}_y \mathbf{x} \right\|_1 \\ 
  &=  \sum\limits^N_{i=1}  \left| (\mathbf{D}_x \mathbf{x})_i \right| + \left| (\mathbf{D}_y \mathbf{x})_i \right| = \sum\limits^N_{i=1} \sqrt[]{(\mathbf{D}_x \mathbf{x})^2_i} + \sum\limits^N_{i=1} \sqrt[]{(\mathbf{D}_y \mathbf{x})^2_i} 
\end{align}$
  
* Isotropic:
$\begin{align} 
  \mathrm{TV}(\mathbf{x}) = \left\| \begin{pmatrix} (\mathbf{D}_x \mathbf{x})_i \\ (\mathbf{D}_y \mathbf{x})_i \end{pmatrix} \right\|_2 = \sum\limits^N_{i=1}  \sqrt[]{(\mathbf{D}_x \mathbf{x})^2_i + (\mathbf{D}_y \mathbf{x})^2_i}
\end{align}$

**TODO: Add images!**

+++

#### Half-quadratic splitting (HQS) method

+++

This section introduces a flexible, powerful and intuitive iterative approach, the half-quadratic splitting (HQS) method, for solving regularized inverse problems formulated like \eqref{eq:map_solution}. 

We start with some general considerations and then apply HQS to the inverse problem of deconvolution.

+++

We assuem the following imge formation model:

$\begin{align} 
  \mathbf{b} = \mathbf{Ax} + \mathbf{\eta},
\end{align}$

with $\mathbf{x}\in \mathbb{R}^{N}$ denoting the unknown vector, $\mathbf{b}\in \mathbb{R}^{M}$ representing the observations, the additive noise $\mathbf{\eta}\in \mathbb{R}^{M}$ and the matrix $\mathbf{A}\in \mathbb{R}^{M\times N}$ encoding the linear image formation model.

+++

Equation \eqref{eq:map_solution} leads us to the general formulation of regularized inverse problems in the field of computational imaging:

\begin{align}\label{eq:general_inverse_problem} 
  \hat{\mathbf{x}} = \argmin{\mathbf{x}} \, \underbrace{\frac{1}{2} \left\| \mathbf{Ax-b} \right\|^2_2}_{\text{data fidelity term}} + \underbrace{\lambda \Psi (\mathbf{x}) }_{\text{regularizer}}\,.
  \end{align}

+++ {"tags": ["book_only"]}

 The data fidelity term ensures that the sought solution $\mathbf{\hat{x}}$ matches the observed data $\mathbf{b}$ when fed through the image formation process (modelled by $\mathbf{A}$). The regularization operator $\Psi : \mathbb{R}^{N} \rightarrow \mathbb{R}^M$ models prior knowledge about the unknown original data $\mathbf{x}$. The scalar parameter $\lambda $ balances between the data fidelity term and the regularization term and hence $\lambda \in [ 0,1 ]$.

+++ {"tags": ["book_only"]}

 
Trying to directly solve \eqref{eq:general_inverse_problem}, e.g., via gradient descent, often does not work well. Reasons are poor convergence or difficulties in finding an efficient way to calculate the gradient. Even worse, whenever we change the regularizer, we will have to re-write the optimization program again.

+++ {"tags": ["presentation_only", "remove-cell"]}

Disadvantages of directly solving \eqref{eq:general_inverse_problem} with, e.g.,  gradient descent:
* Bad / no convergence,
* calculation of gradient only in a computationally inefficient way,
* change of regularizer requires a lot of reprogramming.

+++

Hence, rewrite \eqref{eq:general_inverse_problem} to:

\begin{align} \label{eq:hqs_1}
  \argmin{\mathbf{x}}\quad &\underbrace{\frac{1}{2}\left\| \mathbf{Ax-b} \right\|^2_2 }_{=:f(\mathbf{x})} + \underbrace{\lambda \Psi (\mathbf{z})}_{=:g(\mathbf{z})} \\
  \text{subject to}\quad &\mathbf{Dx-z} = \mathbf{0} \,.
  \end{align}

+++ {"tags": ["book_only"]}

We introduced a so-called *slack variable* $\mathbf{z}\in \mathbb{R}^O$ which allows us to separate the data fidelity term and the regularization term so that they do not depend on the same variable anymore. Obviously, $\mathbf{x}$ and $\mathbf{z}$ are still linked by the constraint $\mathbf{Dx-z=0}$.

For now, we assume $\mathbf{D}\in \mathbb{R}^{N\times O}$ to represent the identity matrix (i.e., it does not introduce any changes and can be ignored for now) - it will come back into play later on.

+++ {"tags": ["presentation_only", "remove-cell"]}

With so-called *slack variable* $\mathbf{z}\in \mathbb{R}^O$.

For now, assume $\mathbf{D}\in \mathbb{R}^{N\times O}$ to represent the identity matrix (will change later).

+++

We now include the constraint of \eqref{eq:hqs_1} directly in the main optimization objective term via a penalty term:

\begin{align}\label{eq:hqs_2}
 L_\rho (\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2\,, \qquad \text{with }\rho > 0\,.
  \end{align}

+++ {"tags": ["book_only"]}

 
 Intuitively, setting $\rho$ to a large value leads to the same results for minimizing \eqref{eq:hqs_1} and \eqref{eq:hqs_2}. The benefit of this reformulation is that we can perform gradient descent for $\mathbf{x}$ and $\mathbf{z}$ in an alternating fashion.

+++ {"tags": ["presentation_only", "remove-cell"]}

 
 $\Rightarrow$ Variables $\mathbf{x}$ and $\mathbf{z}$ can now be iteratively optimized via gradient descent in an alternating fashion:

+++

\begin{align} 
   &\mathbf{x} \leftarrow \mathrm{prox}_{f,\rho} (\mathbf{z}) = \argmin{\mathbf{x}} L_\rho (\mathbf{x}, \mathbf{z}) = \argmin{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2\,, \\
   &\mathbf{z} \leftarrow \mathrm{prox}_{g,\rho} (\mathbf{Dx}) = \argmin{\mathbf{z}} L_\rho (\mathbf{x}, \mathbf{z}) = \argmin{\mathbf{z}} g(\mathbf{z}) + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2\,.
\end{align}

TODO: Explain proximal operators.

+++ {"tags": ["book_only"]}

Again, the important benefit of this formulation is that we can update $\mathbf{x}$ and $\mathbf{z}$ separately. We will see that this approach allows us to easily experiment with different regularizers (this is sometimes also referred to as a *plug-and-play* formulation).

+++ {"tags": ["presentation_only", "remove-cell"]}

Advantages of this formulation:

* Separate updates of $\mathbf{x}$ and $\mathbf{z}$,
* allows to easily experiment with different regularizers,
* many proximal operators can be implemented efficiently, often in closed form.

+++

##### HQS for deconvolution

We now again consider the inverse problem of deconvolution with circual boundary conditions. Here, the matrix $\mathbf{A}$ is the square circulant Toeplitz matrix $\mathbf{C} \in \mathbb{R}^{N\times N}$ representing a 2D-convolution of the input image $x$ with the convolution kernel $c$.

+++

Revise the duality between the signal processing formulation and the algebraic formulation:

\begin{align} 
   c*x = \Fi \left\{ \F \left\{ c \right\} \cdot \F \left\{ x \right\}  \right\} &\Leftrightarrow \mathbf{Cx} \,, \\
   \Fi \left\{ \F \left\{ c \right\}^* \cdot \F \left\{ x \right\}   \right\} &\Leftrightarrow \mathbf{C}\transp \mathbf{x}\,, \\
   \Fi \left\{ \frac{\F \left\{ b \right\} }{\F \left\{ c \right\} } \right\} &\Leftrightarrow \mathbf{C}^{-1} \mathbf{b}\,.
\end{align}

+++

###### HQS with total variation and denoising regularizers

The initial formulation \eqref{eq:hqs_1} of HQS depends on which regularizer we employ.

For total variation this is:
\begin{align} 
  \argmin{\mathbf{x}}\quad &\underbrace{\frac{1}{2}\left\| \mathbf{Dx-b} \right\|^2_2 }_{=:f(\mathbf{x})} + \underbrace{\lambda \left\| \mathbf{z} \right\|_1 }_{=:g(\mathbf{z})} \\
  \text{subject to}\quad &\mathbf{Dx-z} = \mathbf{0} \,,
\end{align}

with $\mathbf{D} = \left( \mathbf{D}\transp_x \mathbf{D}\transp_y \right)\transp \in \mathbb{R}^{2N \times N}$ representing the finite difference operator for calculating the gradients of $\mathbf{x}$ in $x$- and $y$-direction.

+++ {"tags": ["book_only"]}

```{note}
  The vector $\mathbf{z}\in \mathbb{R}^{2N}$ has to be twice as large as $\mathbf{x}\in \mathbb{R}^{N} $ in order to store the two gradient values in $x$- and $y$-direction for every input pixel.
```

+++

In the more general case, we use a regularizer $\Psi $ projecting an image onto the set of feasible natural images (more on that later):

\begin{align} 
  \argmin{\mathbf{x}}\quad &\underbrace{\frac{1}{2}\left\| \mathbf{Dx-b} \right\|^2_2 }_{=:f(\mathbf{x})} + \underbrace{\lambda  \Psi (\mathbf{z})}_{=:g(\mathbf{z})} \\
  \text{subject to}\quad &\mathbf{x-z} = \mathbf{0} \,.
\end{align}

Here the matrix $D$ represents the identity matrix which is why it can be omitted.

+++

###### Efficient implementation of $x$-update

For obtaining the $x$-update, we have to derive the proximal operator $\mathrm{prox}_{f,\rho}$:
\begin{align} 
  \mathrm{prox}_{f,\rho} (\mathbf{z}) = \argmin{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2 = \argmin{\mathbf{x}} \frac{1}{2} \left\| \mathbf{Cx-b} \right\|^2_2 + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2 \,.
\end{align}
Hence, we have to derive the gradient of that equation with respect to $\mathbf{x}$.

+++ {"tags": ["book_only"]}

We iteratively expand that equation as follows:
\begin{align} 
  &\frac{1}{2} \left\| \mathbf{Cx-b} \right\|^2_2 + \frac{\rho}{2} \left\| \mathbf{Dx-z} \right\|^2_2 \\
  =&\frac{1}{2} (\mathbf{Cx-b})\transp (\mathbf{Cx-b}) + \frac{\rho}{2} (\mathbf{Dx-z})\transp (\mathbf{Dx-z}) \\
  =&\frac{1}{2} (\mathbf{x}\transp \mathbf{C}\transp \mathbf{Cx} - 2\mathbf{x}\transp \mathbf{C}\transp \mathbf{b} + \mathbf{b}\transp \mathbf{b}) + \frac{\rho}{2} (\mathbf{x}\transp \mathbf{D}\transp \mathbf{D} \mathbf{x} - 2 \mathbf{x}\transp \mathbf{D}\transp \mathbf{z} + \mathbf{z}\transp\mathbf{z}) \,.
\end{align}

+++

The sought gradient is
\begin{align} 
   \mathbf{C}\transp \mathbf{Cx} - \mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{Dx} - \rho \mathbf{D}\transp \mathbf{z} \,.
\end{align}

This expression can now be equated to zero and solved for $\mathbf{x}$.

+++ {"tags": ["book_only"]}

The single steps are:
\begin{align} 
  \mathbf{C}\transp \mathbf{Cx} - \mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{Dx} - \rho \mathbf{D}\transp \mathbf{z}\quad &\overset{!}{=} \quad \mathbf{0} \\
  \mathbf{C}\transp \mathbf{Cx} + \rho \mathbf{D}\transp \mathbf{Dx}
  &\overset{!}{=} \quad \mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{z} \\
  (\mathbf{C}\transp \mathbf{C} + \rho \mathbf{D}\transp \mathbf{D})\mathbf{x}
  &\overset{!}{=} \quad \mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{z} \\
  \mathbf{x} &\overset{!}{=} (\mathbf{C}\transp \mathbf{C} + \rho \mathbf{D}\transp \mathbf{D})^{-1}(\mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{z}) \,.
\end{align}

+++

This yields a first formulation for the sought proximal operator:
\begin{align}\label{eq:hqs_tv_1} 
  \mathrm{prox}_{f,\rho} (\mathbf{z}) = (\mathbf{C}\transp \mathbf{C} + \rho \mathbf{D}\transp \mathbf{D})^{-1}(\mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{z}) \,.
\end{align}

+++

###### Special case of total variation regularizer

+++ {"tags": ["book_only"]}

For the TV regularizer, the matrix $\mathbf{D}$ represents the finite difference operator.  The matrix-vector multiplications involved in the proximal operator, i.e., $\mathbf{Cx}$ and $\mathbf{Dx} = \left[ \mathbf{D}\transp_x \mathbf{D}\transp_y  \right]\transp \mathbf{x}$ (and also their adjoint correspondencies $\mathbf{C}\transp \mathbf{b}$ and $\mathbf{D}\transp \mathbf{z} = \left[ \mathbf{D}\transp_x \mathbf{D}\transp_y  \right] \mathbf{z} =  \mathbf{D}\transp_x \mathbf{z} + \mathbf{D}\transp_y \mathbf{z}   $ ) all encode 2D-convolutions with circular boundary conditions.

+++

Exploiting the duality of the signal-processing perspective and the algebraic perspective yields the following reforumlations of the denominator, respectively, of the nominator of \eqref{eq:hqs_tv_1}:
\begin{align} 
  (\mathbf{C}\transp \mathbf{C} + \rho \mathbf{D}\transp \mathbf{D}) &\Leftrightarrow \Fi \left\{ \F \left\{ c \right\}^* \cdot \F \left\{ c \right\} + \rho \left( \F \left\{ d_x \right\}^* \cdot \F \left\{ d_x \right\} + \F \left\{ d_y \right\}^* \cdot \F \left\{  d_y \right\}     \right)   \right\} \, , \\
  (\mathbf{C}\transp \mathbf{b} + \rho \mathbf{D}\transp \mathbf{z}) &\Leftrightarrow \Fi \left\{ \F \left\{ c \right\}^* \cdot \F \left\{ b \right\} + \rho \left( \F \left\{ d_x \right\}^* \cdot \F \left\{ z \right\} + \F \left\{ d_y \right\}^* \cdot \F \left\{  z \right\}  	  \right)   \right\} \,,
\end{align}
with $(\cdot)^*$ denoting the element-wise complex conjugate.

+++

Combining both terms in the original fraction yields the sought proximal operator:

\begin{align} 
  \mathrm{prox}_{\left\| \cdot \right\|_2 ,\rho} (\mathbf{z}) = \Fi \left\{  \frac{\F \left\{ c \right\}^* \cdot \F \left\{ b \right\} + \rho \left( \F \left\{ d_x \right\}^* \cdot \F \left\{ z \right\} + \F \left\{ d_y \right\}^* \cdot \F \left\{  z \right\}  	  \right)}{\F \left\{ c \right\}^* \cdot \F \left\{ c \right\} + \rho \left( \F \left\{ d_x \right\}^* \cdot \F \left\{ d_x \right\} + \F \left\{ d_y \right\}^* \cdot \F \left\{  d_y \right\}     \right)} \right\} \,.
\end{align}

+++ {"tags": ["book_only"]}

As for the inverse filer introduced before, this proximal operator is also unstable with respect to noise and zeros in the involved Fourier transforms. However, the integration into the HQS iterations will mitigate these effects so that the resulting estimate $\mathbf{\hat{x}}$ will not be affected.

```{note}
   All terms of the proximal operator that do not depend on $z$ only have to be computed once and can then be reused.
```

+++

###### General regularizer

For a general regularizer, we assume $\mathbf{D}$ to be the identity matrix so that it can be ignored. The proximal operator can then be written as:

\begin{align} 
  \mathrm{prox}_{\left\| \cdot \right\|_2 ,\rho} (\mathbf{z}) = \Fi \left\{ \frac{\F \left\{ c \right\}^* \cdot \F \left\{ b \right\} + \rho \F \left\{ z \right\}  }{\F \left\{ c \right\}^* \cdot \F \left\{ c \right\} + \rho}  \right\} \,.
\end{align}

+++

##### Efficient implementation of $z$-update for anisotropic TV regularizer

For the $z$-update, we need to find a solution for the proximal operator
\begin{align} 
   \mathrm{prox}_{\left\| \cdot \right\|_1, \rho } (\mathbf{Dx}) = \argmin{\mathbf{z}} \lambda \left\| \mathbf{z} \right\|_1 + \frac{\rho}{2}\left\| \mathbf{Dx-z} \right\|^2_2 \,.  
\end{align}

To simplify the writing, we substitute $\mathbf{v} = \mathbf{Dx}$:

\begin{align} 
  \mathrm{prox}_{\left\| \cdot \right\|_1, \rho } (\mathbf{v}) = \argmin{\mathbf{z}} \lambda \left\| \mathbf{z} \right\|_1 + \frac{\rho}{2}\left\| \mathbf{v-z} \right\|^2_2 \,.  
\end{align}

+++ {"tags": ["book_only"]}

We now have to calculate the gradient of that expression with respect to $\mathbf{z}$. To easily follow the derivation, we will work on a single element of the gradient, i.e., we consider the scalar function

\begin{align} 
   h(z) = \frac{\rho}{2}(v-z)^2 + \lambda \vert z \vert \,.
\end{align}

In order to calculate the gradient $h'(z)$ of $h(z)$, we can work on the two terms of the addition separately. The derivative of the first term $\frac{\rho}{2}(v-z)^2$ is 
\begin{align} 
   \frac{\mathrm{d}}{\mathrm{d}z}\,\, \frac{\rho}{2}(v-z)^2 = \rho (-v + z) \,.
\end{align}
Unfortunately, the absolute value function $\vert \cdot \vert$ (i.e., the 1-norm $\left\| \cdot \right\|_1$) is not differentiable and we have to take a detour to solve that problem.

For convex functions there is the concept of subdifferentials and subgradients which can be employed to approximate the gradient of convex functions at positions where they are not differentiable. 

Consider a convex function $f:\mathcal{S} \rightarrow \mathbb{R}$ defined on the open interval $\mathcal{S}$. According to Taylor's theorem, the linear approximation of such a function $f$ at any point is strictly smaller than the value of the function at that point itself, i.e.,
\begin{align} 
   \forall a \in \mathcal{S}: f(x) \geq f(a) + f'(a)(x-a) \,.
\end{align}
This inequality also holds for other values $g$ instead of $f'(a)$, which are smaller than the true gradient $f'(a)$ and since a gradient is never infinite, there have to be infinitely many of suitable $g$:
\begin{align} 
   \forall a \in \mathcal{S} \exists g \in \mathbb{R}: f(x) \geq f(a) + g(x-a) \,.
\end{align}
Such values $g$ are called *subgradients* of the function $f$ at position $a$. The set of all subgradients of $f$ at position $a$ is called the *subdifferential* $\partial_a f$ of $f$ at position $a$:
\begin{align} 
   \partial_a f(x) = \left\{ g\in \mathbb{R}: f(x) \geq f(a) + g(x-a) \right\}  \,.
\end{align}

Subdifferentials have the following useful properties (w.r.t a convex function $f$ as defined above):
* The function $f$ is differentiable at position $a$ if and only if the set of subdifferentials at position $a$ contains only a single slope value, i.e., if $\vert \partial_a f(x) \vert = 1$.
* A point $a$ is a global minimum of $f$ if and only if $0 \in \partial_a f(x)$.
* Let $k(x)$ be another convex function like $f$. With the subdifferentials $\partial_a f(x), \partial_a k(x)$ for some position $a$, the subdifferential of $f+k$ for position $a$ is then $\partial_a (f+k)(x) = \partial_a f(x) \oplus \partial_a k(x)$ with $\oplus$ denoting the so-called Minkowski sum (i.e., the set of all possible sums between all elements of the two input sets).

We will now derive the subdifferential of $h(z) = \frac{\rho}{2}(v-z)^2 + \lambda \vert z \vert $ (neglecting the position $a$ for simplicity) and look for subgradients of $0$, which correspond to the sought minimum.

According to the third property introduced before, it is:
\begin{align} 
   \partial h(z) = \partial \frac{\rho}{2}(v-z)^2 \oplus  \partial \lambda \vert z \vert \,.
\end{align}

Of course, the true gradient $f'(a)$ of a function $f(x)$ at position $a$ is also a valid subgradient at position $a$, i.e.:
\begin{align}\label{eq:hqs_z_1} 
  \partial h(z) = \lbrace \rho(-v+z) \rbrace \oplus  \partial \lambda \vert z \vert \,.   
\end{align}

The subdifferentials for the absolute value function are:
\begin{align}\label{eq:hqs_z_2}
  \partial \lambda \vert z \vert = \begin{cases}
   &-\lambda &\text{ if } z < 0\\
   &[-\lambda, \lambda] &\text{ if } z = 0\\
   &\lambda &\text{ if } z > 0\\
  \end{cases} \,.
\end{align}
For $z\neq 0$, the absolute function is differentiable which is why its subdifferentials for those positions contains only one element ($\lambda$ or $-\lambda$ respectively). For $z=0$ there infinitely many possible slope values inside the interval $[-\lambda, \lambda]$ (note: the value $0$ is also contained in that interval, indicating that the global minimum must be at position $0$). 

Combining \eqref{eq:hqs_z_1} and \eqref{eq:hqs_z_2} yields:
\begin{align}\label{eq:hqs_z_1} 
  \partial h(z) =  \begin{cases}
    &\rho(-v+z) - \lambda &\text{ if } z < 0\\
    &[-\rho v-\lambda, -\rho v + \lambda] &\text{ if } z = 0\\
   \end{cases} \,.
\end{align}

We now equate all cases separately to $0$, solve for $z$ and calculate the corresponding interval for $v$ (which is the variable that we know, so we can use it to decide which case applies):
\begin{align} 
  \textbf{Case 1,  } z < 0: \quad &\rho(-v+z) - \lambda \overset{!}{=}0 \Leftrightarrow z \overset{!}{=} v + \frac{\lambda}{\rho} \\
  &v + \frac{\lambda}{\rho}  \overset{!}{<} 0 \Leftrightarrow v < - \frac{\lambda}{\rho} \,.
\end{align}

For the second case, which is where the sought minimum is located, we want to always return $0$ which is why we have to make sure that $0$ is contained in the interval of possible slopes, i.e.,  $0 \in [-\rho v-\lambda, -\rho v + \lambda]$. Hence, it must hold 
\begin{align}
   -\rho v - \lambda \leq 0 &\Leftrightarrow v \geq - \frac{\lambda }{\rho } \quad \text{and} \\
   0 \leq -\rho v + \lambda  &\Leftrightarrow v \leq \frac{\lambda}{\rho} \,,
\end{align}
i.e., 
\begin{align} 
  \textbf{Case 2,  } z < 0: \quad \left| v \right| \leq \frac{\lambda }{\rho }\,.
\end{align}

The third case can be solved analogously to the first case:
\begin{align} 
  \textbf{Case 3,  } z > 0: \quad &\rho(-v-z) + \lambda \overset{!}{=}0 \Leftrightarrow z \overset{!}{=} v - \frac{\lambda}{\rho} \\
  & v - \frac{\lambda}{\rho}  \overset{!}{>} 0 \Leftrightarrow v > \frac{\lambda}{\rho} \,.
\end{align}
$\square$

+++

The sought proximal operator is given by
\begin{align} 
  \mathrm{prox}_{\left\| \cdot \right\|_1, \rho } (\mathbf{v})_i =  \mathcal{S}_{\lambda / \rho} (v_i) = \begin{cases}
  v_i + \frac{\lambda}{\rho} &\text{if } v_i < - \frac{\lambda}{\rho}\\
  0 &\text{if } \left| v_i \right| \leq \frac{\lambda }{\rho }\\
  v_i - \frac{\lambda}{\rho} &\text{if } v_i > \frac{\lambda}{\rho} 
\end{cases} \,,  
\end{align}
with $v_i$ refering to the $i$-th element of the corresponding vector $\mathbf{v}$ (etc.). 

The operator $\mathcal{S}$ is also called the element-wise *soft thresholding operator*.

+++

##### Visualization of soft thresholding operator

```{code-cell} ipython3
def y(z,v, rho, lam):
    return lam*np.abs(z) + rho/2.0 * np.linalg.norm(v-z)**2
```

```{code-cell} ipython3
def s(v,kap):
    if v > kap:
        return v-kap
    if v < (-1*kap):
        return v+kap
    else:
        return 0
```

```{code-cell} ipython3
def soft_thres_visu(v):
    rho = 0.5
    lam = 0.4

    zs = np.arange(-5.0,5.0,0.1)
    ys = [y(z,v,rho,lam) for z in zs]

    vs = np.arange(-3.0,3.0,0.1)
    ss = [s(v,lam/rho) for v in vs]

    #plt.subplots(1,2)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(zs, ys)
    plt.plot(cur_s:=s(v, lam/rho), y(cur_s, v, rho, lam), 'bo' )

    plt.subplot(1,2,2)
    plt.plot(vs, ss)
    plt.plot(v,s(v,lam/rho),'bo')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.subplots(1,2)
interact(lambda v: soft_thres_visu(v), v=widgets.FloatSlider(min=-3.0,max=3.0, step=0.2, value=0))
```

##### Efficient implementation of $z$-update for isotropic TV regularizer

+++

For the isotropic case, the $l_2$-norm of the finite differences approximation of the horizontal and vertical image gradients is used as the regularizer instead of the $l_1$-norm, i.e.,
\begin{align} 
   \lambda \left\| \mathbf{z} \right\|_{2,1}  = \lambda \sum\limits^N_{i=1} \left\| \left[ \begin{pmatrix} 
      (\mathbf{D}_x \mathbf{x})_i \\ (\mathbf{D}_y \mathbf{x})_i
   \end{pmatrix} \right] \right\|_2 = \lambda \sum\limits^N_{i=1} \sqrt{(\mathbf{D}_x \mathbf{x})^2_i + (\mathbf{D}_y \mathbf{x})^2_i } \,.
\end{align}
This expression is also called the *group lasso*.

+++

Accordingly, the whole deconvolution problem for the isotropic TV regularizer is
\begin{align} 
   &\argmin{\mathbf{x}} \underbrace{\frac{1}{2} \left\| \mathbf{Cx-b} \right\|^2_2}_{=:f(\mathbf{x})} + \underbrace{\lambda \sum\limits^N_{i=1} \left\| \left[ \begin{pmatrix} 
      z_i \\ z_{i+N}
   \end{pmatrix} \right]  \right\|_2 }_{=:g(\mathbf{z})} \\
   &\text{subject to } \mathbf{Dx-z=\mathbf{0}}\,.
\end{align}

+++ {"tags": ["book_only"]}

Again, the vector $\mathbf{z} \in \mathbb{R}^{2\times N}$ so that its first, respectively, second half can hold the image gradients in $x$-direction, respectively, in $y$-direction, i.e., $\mathbf{z} = \left[ \mathbf{D}_x \mathbf{x}\,\, \mathbf{D}_y \mathbf{x}   \right]\transp $.

+++

As only the regularization term $g(\mathbf{z})$ has changed, we only have to derive a corresponding $\mathbf{z}$-update rule, i.e., we have to find a solution for the proximal operator
\begin{align} 
   \mathbf{z} \leftarrow \mathbf{prox}_{\left\| \cdot \right\|_{2,1},\rho } (\mathbf{v}) = \argmin{\mathbf{z}}\,\, \lambda \sum\limits^N_{i=1} \left\| \left[ \begin{pmatrix} 
    z_i \\ z_{i+N}
 \end{pmatrix} \right]  \right\|_2  + \frac{\rho}{2} \left\| \mathbf{v-z} \right\| ^2_2, \quad \mathbf{v=Dx} \,.
\end{align}

+++ {"tags": ["book_only"]}

Therefore, we again extract the term we have to minimize:
\begin{align}\label{eq:hqs:iso:1}
   h(\mathbf{z}) := \lambda \left\| \mathbf{z} \right\|_2 + \frac{\rho}{2} \left\| \mathbf{v-z} \right\| ^2_2 \,.
\end{align}

We follow the approach of calculating the gradient of $h(\mathbf{z})$, equating it to $\mathbf{0}$ and solving for $\mathbf{z}$.

The gradient of the second term with respect to $\mathbf{z}$ can be calculated straight forward:
\begin{align} 
   \nabla_\mathbf{z} \frac{\rho}{2} \left\| \mathbf{v-z} \right\| ^2_2 = \rho (-\mathbf{v} + \mathbf{z}) \,.
\end{align}

The gradient of the Euclidean norm $\left\| \mathbf{x} \right\| _2$ is
\begin{align} 
   \nabla_\mathbf{x} \left\| \mathbf{x} \right\|_2 = \frac{\mathbf{x}}{\left\| \mathbf{x} \right\|_2 }, \quad \text{for } \mathbf{x} \neq \mathbf{0} \,.
\end{align}
Hence, the gradient of the first term of \eqref{eq:hqs:iso:1} is 
\begin{align} 
   \nabla_\mathbf{z} \lambda \left\| \mathbf{z} \right\| _2 = \lambda \frac{\mathbf{z}}{\left\| \mathbf{z} \right\|_2 } \quad \text{for } \mathbf{z} \neq \mathbf{0} \,.
\end{align}

So for $\mathbf{z} \neq \mathbf{0}$, we will find a $\hat{\mathbf{z}}$ with $  \nabla_\mathbf{z} \lbrace h(\mathbf{z}) \rbrace (\hat{\mathbf{z}}) = \mathbf{0}$:
\begin{align} 
  \lambda \frac{\mathbf{z}}{\left\| \mathbf{z} \right\|_2 } +  \rho (-\mathbf{v} + \mathbf{z}) &\overset{!}{=} \mathbf{0} \\
  \mathbf{z}\left( \frac{\lambda}{\left\| \mathbf{z} \right\|_2 } + \rho \right) &= \rho \mathbf{v} \,. \label{eq:hqs:iso:2}
\end{align}
We now apply the $l_2$-norm to both sides of the equation:
\begin{align}  \label{eq:hqs:iso:3}
   \left| \frac{\lambda}{\left\| \mathbf{z} \right\|_2 }  + \rho\right| \cdot \left\| \mathbf{z} \right\|_2 = \left| \rho \right|  \left\| \mathbf{v} \right\|_2 \,.
\end{align}
The last step introduced two absolute values for each of which the two possible cases, .i.e, $>0$, $<0$ have to be considered. Regarding the term $\left| \rho \right| $ on the right side of the equation, only the positive case has to be considered since $\rho > 0$ by definition.

The first absolute term $\left| \frac{\lambda}{\left\| \mathbf{z} \right\|_2 }  + \rho\right|$ is positive if
\begin{align} 
   \frac{\lambda}{\left\| \mathbf{z} \right\|_2 }  + \rho &> 0 \\
   - \frac{\lambda}{\rho} &< \left\| \mathbf{z} \right\| _2 \,,
\end{align}
what always holds since due to $\lambda > 0$ and $\rho > 0$ it follows $-\frac{\lambda}{\rho} < 0$ and since  $\left\| \mathbf{z} \right\| _2 > 0$ as $\left\| \cdot \right\| \geq 0$ for any norm and $\mathbf{z} \neq \mathbf{0}$ by definition of the considered case. Hence, the negative case is impossible.

We continue to solve \eqref{eq:hqs:iso:3} for $\left\| \mathbf{z} \right\| _2$:
\begin{align} 
  \left( \frac{\lambda}{\left\| \mathbf{z} \right\|_2 }  + \rho\right) \cdot \left\| \mathbf{z} \right\|_2 &= \rho  \left\| \mathbf{v} \right\|_2  \\
  \lambda + \rho \left\| \mathbf{z} \right\| _2 &= \rho \left\| \mathbf{v} \right\| _2 \\
  \left\| \mathbf{z} \right\| _2 &= \frac{\rho \left\| \mathbf{v} \right\|_2 - \lambda  }{\rho} \,,
\end{align}
and now insert this expression for $\left\| \mathbf{z} \right\| _2$ into \eqref{eq:hqs:iso:2} and solve for $\mathbf{z}$:
\begin{align} 
   \mathbf{z} \left( \lambda \cdot \frac{\rho }{\rho \left\| \mathbf{v} \right\|_2 - \lambda } + \rho  \right) &= \rho \mathbf{v} \\
   \mathbf{z} \frac{\lambda + \rho  \left\| \mathbf{v} \right\|_2 - \lambda  }{\rho  \left\| \mathbf{v} \right\|_2 - \lambda  } &= \mathbf{v}  \\
   \mathbf{z} &= \mathbf{v} \cdot \frac{\rho  \left\| \mathbf{v} \right\|_2 - \lambda  }{\rho  \left\| \mathbf{v} \right\|_2 } \\
   \mathbf{z} &= \mathbf{v} \cdot \left( 1 - \frac{\lambda }{\rho \left\| \mathbf{v} \right\|_2 } \right) \,.
\end{align}
In order to make sure that $\mathbf{z} \overset{!}{\neq} \mathbf{0}$, we have to check for which conditions $\left( 1 - \frac{\lambda }{\rho \left\| \mathbf{v} \right\|_2 } \right) > 0$ holds (we do not have to check for $<0$ as this is impossible as we showed when inspecting the absolute terms before):
\begin{align} 
  \left( 1 - \frac{\lambda }{\rho \left\| \mathbf{v} \right\|_2 } \right) &> 0 \\
  \left\| \mathbf{v} \right\|_2 &> \frac{\lambda }{\rho } \,.
\end{align}

For $\mathbf{z} = \mathbf{0}$, there is no well-defined gradient for $\left\| \mathbf{z} \right\|_2$, so we make use of the concept of subdifferentials again and derive the subdifferential of $h(\mathbf{z})$ with respect to position $\mathbf{z} = \mathbf{0}$:
\begin{align} 
  h(\mathbf{z}) &= \lambda \left\| \mathbf{z} \right\|_2 + \frac{\rho}{2} \left\| \mathbf{v-z} \right\| ^2_2 \\
  \frac{h(\mathbf{z})}{\rho} &= \frac{\lambda}{\rho } \left\| \mathbf{z} \right\|_2 + \frac{1}{2} \left\| \mathbf{v-z} \right\|^2_2  \\
  \partial_\mathbf{0} \frac{h(\mathbf{z})}{\rho} &= \partial_\mathbf{0} \frac{\lambda}{\rho } \left\| \mathbf{z} \right\|_2 \oplus  \partial_\mathbf{0}\frac{1}{2} \left\| \mathbf{v-z} \right\|^2_2 \,.
\end{align}
As before, we can replace the second subdifferential on the right side of the equation with the true gradient:
\begin{align} 
  \partial_\mathbf{0} \frac{h(\mathbf{z})}{\rho} &= \partial_\mathbf{0} \frac{\lambda}{\rho } \left\| \mathbf{z} \right\|_2 \oplus  \lbrace - \mathbf{v} \rbrace \,.
\end{align}
For the first term, i.e., $\partial_\mathbf{0}\frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2$, we employ the definition of subdifferentials:
\begin{align} 
  \partial_\mathbf{0}\frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 &= \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 \geq \frac{\lambda}{\rho} \left\| \mathbf{0} \right\|_2 + \left\langle \mathbf{g} , \mathbf{z} - \mathbf{0}\right\rangle  \right\} \\
  \partial_\mathbf{0}\frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 &= \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 \geq \left\langle \mathbf{g} , \mathbf{z}\right\rangle  \right\} \,.
\end{align}
By means of the Cauchy-Schwarz inequality, i.e., $\left\langle \mathbf{g}, \mathbf{z} \right\rangle \leq \left\| \mathbf{g} \right\| \cdot \left\| \mathbf{z} \right\| $, we can further simplify the derivation:
\begin{align} 
  \partial_\mathbf{0}\frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 &= \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 \geq \left\langle \mathbf{g} , \mathbf{z}\right\rangle  \right\} \\
  &= \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho} \left\| \mathbf{z} \right\|_2 \geq \left\| \mathbf{g} \right\|_2 \cdot \left\| \mathbf{z} \right\|_2     \right\} \\
  &= \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho}  \geq \left\| \mathbf{g} \right\|_2      \right\} \,.
\end{align}
The resulting combined expression for the subdifferential of $\frac{h(\mathbf{z})}{\rho} $ is:
\begin{align} 
   \partial_\mathbf{0}\frac{h(\mathbf{z})}{\rho} = \left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho}  \geq \left\| \mathbf{g} \right\|_2      \right\} \oplus \lbrace -\mathbf{v} \rbrace \,.
\end{align} 
For $\mathbf{z} = \mathbf{0}$ we want to return $\mathbf{0}$ if possible since that would correspond to the sought minimum. A subgradient $\mathbf{0}$ is contained in $\partial_\mathbf{0}\frac{h(\mathbf{z})}{\rho}$ if the first subdifferential $\left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho}  \geq \left\| \mathbf{g} \right\|_2      \right\}$ contains a vector $\mathbf{g}$ with $\mathbf{g} = \mathbf{v}$ because then the Minkowski sum $\left\{ \mathbf{g} \in \mathbb{R}^{2N} : \frac{\lambda}{\rho}  \geq \left\| \mathbf{g} \right\|_2      \right\} \oplus \lbrace -\mathbf{v} \rbrace$ would contain $\mathbf{0}$. This is only possible for any vectors $\mathbf{v}$ with $\left\| \mathbf{v} \right\|_2 \leq \frac{\lambda}{\rho}  $.

With these results, the final $\mathbf{z}$-update rule is given by:

+++ {"tags": ["presentation_only", "remove-cell"]}

The concept of subdifferentials yields the sought $\mathbf{z}$-update rule:

+++

\begin{align} 
  \mathbf{z} \leftarrow \mathbf{prox}_{\left\| \cdot \right\|_{2,1},\rho } (\mathbf{v}) = \begin{cases} 
    \mathbf{v} \cdot \left( 1 - \frac{\lambda }{\rho \left\| \mathbf{v} \right\|_2 } \right) &\text{if } \left\| \mathbf{v} \right\| _2 > \frac{\lambda}{\rho} \\
    \mathbf{0} &\text{if } \left\| \mathbf{v} \right\| _2 \leq \frac{\lambda}{\rho}
  \end{cases}
   \,.
\end{align}
This expression is also known as the vector soft-thresholding operator.
