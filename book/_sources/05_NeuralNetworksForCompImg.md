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
                  if (tags[j]=="book_only" | tags[j]=="remove-cell") {cur_cell.element.hide();}
                  if (tags[j]=="presentation_only") {cur_cell.element.show();}
            }}}
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
%matplotlib notebook
book = False
```

```{code-cell} ipython3
:tags: [remove-input, book_only]

%matplotlib inline
book = True
```

```{code-cell} ipython3
---
init_cell: true
slideshow:
  slide_type: notes
tags: [remove-cell]
---
def showFig2(path,i,ending, width, height):
    imgToShow = plt.imread(f"{path}{i}{ending}")
    plt.imshow(imgToShow)
```

```{code-cell} ipython3
---
init_cell: true
slideshow:
  slide_type: notes
tags: [remove-cell]
---
def showFig(path,i,ending, width, height):
    filename = path+str(i)+ending
    return HTML("<img src=\"" + filename +  f"\" style=\"max-height:{height}vh\"/>")
```

```{code-cell} ipython3
---
init_cell: true
slideshow:
  slide_type: notes
tags: [remove-cell]
---
def imshow(img, cmap=None):
    plt.close('all')
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()
```

```{code-cell} ipython3
---
init_cell: true
slideshow:
  slide_type: notes
tags: [remove-cell]
---
def imshow2(img, cmap=None):
    #plt.close('all')
    #plt.figure()
    plt.clf()
    plt.imshow(img, cmap=cmap)
    #plt.show()
```

```{code-cell} ipython3
interact(lambda i: showFig('figures/4/desk_lightsources_example_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=5), step=1, value=(max_i if book else min_i)))
```

```{code-cell} ipython3
<img src="figures/4/light_path_categories.png" style="max-width:30vw">
```

+++ {"slideshow": {"slide_type": "slide"}}

$\begin{align}
  \newcommand{transp}{^\intercal}
  \newcommand{F}{\mathcal{F}}
  \newcommand{Fi}{\mathcal{F}^{-1}}
  \newcommand{inv}{^{-1}}
  \newcommand{stochvec}[1]{\mathbf{\tilde{#1}}}
  \newcommand{argmax}[1]{\underset{#1}{\mathrm{arg\, max}}}
  \newcommand{argmin}[1]{\underset{#1}{\mathrm{arg\, min}}}
\end{align}$

<font size="7"> Computational Imaging </font><br><br><br>

+++ {"slideshow": {"slide_type": "fragment"}}

# Neural Networks for Computational Imaging

+++ {"slideshow": {"slide_type": "subslide"}}

##### Content
* Introduction to neural networks
* Principal building blocks of a neural network
* Universal approximation theorem
* Gradient descent
* Automatic differentiation
* Architectures and loss functions
* Regularization
* Libraries, tools and other resources

+++

## Introduction to neural networks

* (Artificial) neural networks represent a class of machine learning methods.
* They can be imagined as versatile approximators $\varphi_\boldsymbol{\theta}$ of arbitrary, continuous functions $\mathbf{y} = f(\mathbf{x}), \quad f:\mathbb{R}^N \mapsto \mathbb{R}^M,\quad M,N \in \mathbb{N}$, i.e., with $\varphi_\boldsymbol{\theta}(\mathbf{x}) \approx f(\mathbf{x})$.
* They are defined by their architecture and the corresponding parameters $\boldsymbol{\theta}$.

+++

* Via a suitable training procedure and a so-called *training set* of example pairs $\mathcal{T} = \left\{ (\mathbf{x}_i, \mathbf{y}_i), i \in \left[ 1,\ldots , N \right]  \right\}$ of input variables $\mathbf{x}_i$ and corresponding output variables $\mathbf{y}_i$, their parameters $\boldsymbol{\theta}$ are optimized so that 
  * $\forall (\mathbf{x},\mathbf{y}) \in \mathcal{T}:\text{dist}(\varphi_\boldsymbol{\theta}(\mathbf{x}), \mathbf{y} ) \rightarrow \text{Min.}\,,$ with a suitable distance function $\text{dist}$ and
  * (hopefully) $\text{dist}(\varphi_\boldsymbol{\theta}(\mathbf{x}), f(\mathbf{x}) ) \rightarrow \text{Min.}\,,$ for unseen input vectors $\mathbf{x}$, i.e., which are not part of the training set.

+++

* Neural networks have first been described in 1943 by Warren McCulloch and Walter Pitts in their paper "A Logical Calculus of the Ideas Immanent in Nervous Activity".
* Frank Rosenblatt followed their approach and described the so-called *Perceptron* as a fundamental unit of early neural networks.

+++

* Approximately around the year 2010, researchers started to use very deep neural networks, i.e., with many so-called *layers* (more information later) and achieved unprecedented performances on various tasks in the field of machine learning and computer vision.

+++

* An important enabler for this breakthrough were the increase in computing power provided by modern computers, especially by GPUs (graphics processing units), and the availability and usage of huge amounts of training data.

+++

## Principal building blocks

The two fundamental building blocks of neural networks are
* Matrix vector multiplications and
* non-linear functions, also called or *activation functions*.

Multiple instances of these building blocks can be stacked in parallel or consecutively with respect to each other to finally yield a neural network.

+++

### Layers

When stacking linear or non-linear building blocks in parallel, the resulting structure is called a linear, respectively, a non-linear layer (usually either a linear or a non-linear block is stacked in parallel, not a mixture of both).

+++

The way of stacking the individual blocks is called the *architecture* of the neural network.

+++

### Linear layers

In one building block of a linear layer, the input $(x_1, x_2, \ldots, x_N)\transp$ is mapped to the scalar output $y$ via a linear transformation, i.e.,

$\begin{align} 
   y = \sum\limits^{N}_{i=1} w_i \cdot x_i + b \,,
\end{align}$

with $w_i$ denoting the $i$-th so-called *weight*, i.e. parameter, of this block and $b$ denoting the so-called *bias* (also a parameter), i.e., an additive term not depending on the input.

+++

When one input is simultaneously processed by $K$ linear blocks, i.e.,

$\begin{align} 
   y_k = \sum\limits^{N}_{i=1} w^k_i \cdot x_i + b^k = \underbrace{\left( w^k_1, w^k_2, \ldots, w^k_N, b^k \right)}_{\mathbf{w}\transp_k} \cdot \underbrace{\begin{pmatrix} 
      x_1 \\ x_2 \\ \vdots \\ x_N \\ 1
   \end{pmatrix}}_{\mathbf{x}}  \,,
\end{align}$

for block $k$, this can be expressed compactly via matrix-vector multiplications:

$\begin{align} 
   \begin{pmatrix} 
      y_1 \\ y_2 \\ \vdots \\ y_K
   \end{pmatrix}
   &= 
   \begin{pmatrix} 
      \qquad \mathbf{w}\transp_1 \qquad  \\ \mathbf{w}\transp_2 \\ \vdots \\ \mathbf{w}\transp_K
   \end{pmatrix} \cdot 
   \begin{pmatrix} 
      x_1 \\ x_2 \\ \vdots \\ x_N \\ 1
   \end{pmatrix} \\
   &= \qquad \quad \mathbf{W} \quad \qquad \cdot \quad  \mathbf{x} \,,
\end{align}$

with $\mathbf{W} \in \mathbb{R}^{K \times (N+1)}$ and $\mathbf{x} \in \mathbb{R}^{(N+1) \times 1}$.

+++

This expression can be further extended for the case when multiple input vectors $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_B$, i.e., a so-called *batch* of size $B$, have to be processed simultaneously:

$\begin{align} 
  \begin{pmatrix} 
    \qquad \mathbf{w}\transp_1 \qquad  \\ \mathbf{w}\transp_2 \\ \vdots \\ \mathbf{w}\transp_K
 \end{pmatrix} \cdot 
 \begin{pmatrix} 
     \\  \\ \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_B    \\ \\ \\
 \end{pmatrix} &= \begin{pmatrix} 
    \mathbf{w}\transp_1 \cdot \mathbf{x}_1 & \mathbf{w}\transp_1 \cdot \mathbf{x}_2 &\cdots &\mathbf{w}\transp_1 \cdot \mathbf{x}_B \\ 
    \mathbf{w}\transp_2 \cdot \mathbf{x}_1 & \mathbf{w}\transp_2 \cdot \mathbf{x}_2 &\cdots &\mathbf{w}\transp_2 \cdot \mathbf{x}_B \\     
    \vdots & \vdots & \ddots & \vdots \\
    \mathbf{w}\transp_K \cdot \mathbf{x}_1 & \mathbf{w}\transp_K \cdot \mathbf{x}_2 &\cdots &\mathbf{w}\transp_K \cdot \mathbf{x}_B \\ 
 \end{pmatrix}\\
 &= \begin{pmatrix} 
  \\  \\ \mathbf{y}_1 & \mathbf{y}_2 & \cdots & \mathbf{y}_B    \\ \\ \\
\end{pmatrix}
 \,.
\end{align}$

### Non-linear layers

Neural networks constructed only out of linear layers are very limited in their approximation abilities since in essence they just represent a long linear function and hence can only mimic linear functions.

This is why additional, so-called *non-linear layers* consisting of non-linear building blocks are necessary.

+++

In general, a non-linear building block is a non-linear function $\psi:\mathbb{R}\rightarrow \mathbb{R}$ that is applied to the scalar output of a linear building block.

+++

A popular example for a non-linearity is the so-called *sigmoid*-function 

$\begin{align} 
  \psi(y)=\frac{1}{1+\mathrm{e}^{-y} } \,.
\end{align}$

+++

When choosing high values for $\mathbf{W}$, the sigmoid-function resembles a unit step-function which can be shifted left or right by adjusting the bias $b$:

```{code-cell} ipython3
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def plot_sigmoid_after_linear(a,b):
    xs = np.linspace(0,1,1000)
    y1s = xs*a + b
    y2s = sigmoid(y1s)
    plt.clf()
    plt.plot(xs,y2s)
```

```{code-cell} ipython3
plt.figure()
interact(lambda w,b: plot_sigmoid_after_linear(w,b), w=widgets.FloatSlider(min=1,max=200, step=1, value=8), b=widgets.FloatSlider(min=-100,max=100, step=1, value=-4))
```

In batch processing, he result of a batch of data processed by a linear layer, i.e. $\mathbf{y} = \mathbf{Wx}$, is processed by the non-linearity in an element-wise fashion.

+++

## Universal approximation theorem

+++

It could be shown, that a neural network consisting only of one (sufficiently large) linear layer and one non-linear layer which are combined by a single linear building block can approximate any continuous function. 

TODO: Add figure.

In the following, we will sketch the proof of that theorem.

+++

Consider again the sigmoid function $\psi$ from before applied to a linear block, i.e., $\psi(wx+b)$  form before. 

The position of the unit step approximated by $\psi$ for high $w$ resides at the position $s=-\frac{b}{w}$. Since this is easier to interpret, we will focus on the parameter $s$ from now on.

+++

We now consider two of such blocks added together by an additional single linear building block, i.e.,

$\begin{align} 
   \psi_{s_1} (x)\cdot w_1 + \psi_{s_2}(x)\cdot w_2  + b
\end{align}$

with the respective positions $s_1, s_2$ of the step functions.

```{code-cell} ipython3
def linear(x, s):
    w = 1000
    b = -1 * w * s
    return x*w + b
def plot_2_neurons(s1, s2, w1, w2):
    xs = np.linspace(0,1,1000)
    y1s = sigmoid(linear(xs, s1))
    y2s = sigmoid(linear(xs, s2))
    res = w1 * y1s + w2 * y2s
    plt.clf()
    plt.plot(xs,res)
plt.figure()
interact(lambda s1, s2, w1, w2: plot_2_neurons(s1, s2, w1, w2), \
         s1 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.2), \
         s2 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.6), \
         w1 = widgets.FloatSlider(min=-2,max=2, step=0.1, value=0.4), \
         w2 = widgets.FloatSlider(min=-2,max=2, step=0.1, value=0.6))
```

As can be seen, this addition yields to consecutive step functions what can be used, e.g., to approximate the $\mathrm{rect}$-function.

Therefore, if $s_1 < s_2$, it must hold $w_2 = -w_1$ to get a $\mathrm{rect}$-function with height $h=\left| w_1 \right| = \left| w_2 \right|  $.

+++

We can now add two of such pairs of blocks together to model two $\mathrm{rect}$-functions, i.e., with start, stop positions $s_{1,1}, s_{1,2}$ and height $h_1$ of the first $\mathrm{rect}$-function and $s_{2,1}, s_{2,2}, h_2$ for the second one.

```{code-cell} ipython3
def rect_approx(x, s1, s2, h):
    w1 = h
    w2 = -1 * h
    y1s = sigmoid(linear(x, s1))
    y2s = sigmoid(linear(x, s2))
    return w1 * y1s + w2 * y2s
    
def plot_2_rects(s11, s12, s21, s22, h1, h2):
    xs = np.linspace(0,1,1000)
    y1s = rect_approx(xs, s11, s12, h1)
    y2s = rect_approx(xs, s21, s22, h2)
    res = y1s + y2s
    plt.clf()
    plt.plot(xs,res)
plt.figure()
interact(lambda s11, s12, s21, s22, h1, h2: plot_2_rects(s11, s12, s21, s22, h1, h2), \
         s11 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.2), \
         s12 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.3), \
         s21 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.4), \
         s22 = widgets.FloatSlider(min=0.0,max=1.0, step=0.1, value=0.6), \
         h1 = widgets.FloatSlider(min=-2,max=2, step=0.1, value=0.3), \
         h2 = widgets.FloatSlider(min=-2,max=2, step=0.1, value=-0.4))
```

The more of these modules we add, the more complicated the shape of the output can be. When the width of the single $\mathrm{rect}$-functions approaches zero and the number of $\mathrm{rect}$-functions approaches infinity, any continuous function can be approximated.

This also holds for higher dimensions.

```{code-cell} ipython3
xs = np.linspace(-0.2,1.2,1000)
s1s = np.array([0, 0.2, 0.4, 0.6, 0.8])
s2s = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
hs = np.array([0.1, 0.7, 0.1, -0.1, -0.8])
res = np.zeros_like(xs)

for i in range(0, s1s.shape[0]):
    res = res + rect_approx(xs, s1s[i], s2s[i], hs[i])

plt.figure()
plt.plot(xs,res)
```

Although the theorem states that such a simple structure is enough to approximate any function, in practice network structures with more but thinner layers (i.e., with fewer building blocks) are employed as they can achieve similar results with notably fewer building blocks.

+++

## Optimization of network parameters

How can we automatically determine the parameters $\boldsymbol{\theta}$ of the network so that $\forall (\mathbf{x},\mathbf{y}) \in \mathcal{T}:\text{dist}(\varphi_\boldsymbol{\theta}(\mathbf{x}), \mathbf{y} ) \rightarrow \text{Min.}\,$?

+++

### Gradient descent

+++

To find this minimum, we compute the gradient $\nabla\text{dist}_\boldsymbol{\theta}$ of the distance function $\text{dist}$ with respect to the network parameters $\boldsymbol{\theta}$.

We can then iteratively update an initial guess $\hat{\boldsymbol{\theta}}_0$ (e.g., random) of the network parameters by pushing it into the inverse direction of the gradient $\nabla\text{dist}_\boldsymbol{\theta}$, i.e., into the direction of the nearest minimum:

$\begin{align} 
   \hat{\boldsymbol{\theta}}_{i+1} \leftarrow \hat{\boldsymbol{\theta}}_{i} - \eta \nabla\text{dist}_\boldsymbol{\theta}(\varphi_{\hat{\boldsymbol{\theta}_i}}(\mathbf{X}), \mathbf{Y} )\,, (\mathbf{X},\mathbf{Y}) \in \mathcal{T}
\end{align}$

with $\mathbf{X}, \mathbf{Y}$ indicating batches of multiple training vectors $(\mathbf{x}, \mathbf{y}) \in \mathcal{T}$ and $\eta$ denoting the so-called *learning rate* or the *step size* for the gradient descent updates.

+++

* In every gradient descent iteration, the parameters are updated with regard to the respective batch of training samples $(\mathbf{X}, \mathbf{Y})$ chosen in that iteration.
* In practical scenarios it is usually not possible to process the whole training set in one gradient descent step as the respective data would not fit into the available memory. 
* Hence, in every iteration another batch of training data is used, so that eventually all training samples have been used. The partitioning of $\mathcal{T}$ into those batches is usually performed randomly. This is why this kind of gradient descent is sometimes referred to as *stochastic gradient descent*.
* The set of gradient descent iterations needed to cycle through all training data once is a so-called *epoch*.

+++



* Hence, only a subset of the 
