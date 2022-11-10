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

# Light Transport Analysis

+++ {"slideshow": {"slide_type": "subslide"}}

##### Content
* Introduction into light transport
    * The light transport equation
    * The light transport matrix
* Optical computing for fast light transport 
    * Optical interpretation of matrix vector products
    * Krylov subspace methods (optical power iteration, optical Arnoldi)
* Light transport matrix processing for visual inspection
    * Light transport matrix feature extraction
    * Signal-to-noise ratio based image fusion
* Separation of direct and indirect light transport
    * Inverse light transport
    * Indirect appearance by structured light transport
* Compressed sensing for light transport analysis
    * Introduction to compressed sensing
    * Compressive light transport matrix acquisition

+++ {"slideshow": {"slide_type": "slide"}}

## Introduction into light transport

+++

* Light transport = all light travelling in a scene from illumination sources to sensor elements (i.e., pixels).
* Can be mathematically described via the so-called *light transport matrix* $\mathbf{T}$, if the common assumption of linear light transport is made.

+++

The matrix $\mathbf{T}$ contains the individual influences of all $P$ light sources to all $I$ sensor elements and hence has dimensions of $\mathbf{T} \in \mathbb{R}^{I\times P}_+$.

+++ {"tags": ["book_only"]}

The elements of $\mathbf{T}$ are non-negative as only positive energy or no energy can be transported via light.

+++

The *light transport equation* 
$\begin{align} 
   \mathbf{i} = \mathbf{T}\mathbf{p}
\end{align}$
describes the formation of the sensor values $\mathbf{i}$ of the $I$ sensor elements when the scene is illuminated by the $P$ light sources with respective intensities $\mathbf{p}$.

+++

For the following content, we will assume illumination of the scene by a projector projecting a two-dimensional image and a two-dimensional gray-value camera serving as the sensor.

+++

In this case, $\mathbf{i}$ and $\mathbf{p}$ are the vector representations of the camera image, respectively, of the projected image or pattern.

+++

If the light transport matrix $\mathbf{T}$ is known, the camera image that would result from the illumination with an arbitrary pattern $\mathbf{p}$ could be synthetically calculated by just evaluating the light transport equation
$\begin{align} 
  \mathbf{i} = \mathbf{T}\mathbf{p} \,.
\end{align}$

+++

##### Example

```{code-cell} ipython3
interact(lambda i: showFig('figures/4/desk_lightsources_example_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=5), step=1, value=(max_i if book else min_i)))
```

(All images in this chapter are kindly provided by Matthew O'Toole).

+++

In this sense, the element $T[m,n]$ of $\mathbf{T}$ encodes the contribution of light source (i.e., projector pixel) $n$ to camera pixel $m$.

+++

Unfortunately, the matrix $\mathbf{T}$ is too large to measure and handle digitally for many practically relevant applications.

+++

### Practical limitations 

Consider a pair of a camera and a projector both having a (comparatively low) resolution of 1 mega pixels, then the resulting light transport matrix would have $10^{12}$ elements. 

* With an acquisition rate of $30$ Hz, acquiring the required $10^6$ measurements would need over 9 hours of time.
* For a quantization of 8 bit per pixel for the camera images, $\mathbf{T}$ would require about one terabyte of storage.

+++

### Optical linear algebra

To overcome the mentioned limitations, we will study approaches to perform linear algebra calculations involving $\mathbf{T}$ only in the optical domain, i.e., without ever capturing the single elements of $\mathbf{T}$.

+++

### Light paths

We again assume that all geometric structures involved in our imaging system are significantly larger than the wavelength of the employed light, so that the model of geometric optics, i.e., of light propagating along rays, is valid.

+++

While travelling through the scene, light rays can be refracted, reflected, scattered, etc. until they either hit one of the camera's pixels or get absorbed or exit the scene.

+++

We denote a sequence of such light rays which connect a light source (i.e., a projector pixel) and a camera pixel as a *light path*.

+++

Light paths can be categorized into *direct* and *indirect* paths.

+++

<img src="figures/4/light_path_categories.png" style="max-width:30vw">

+++

* **Black ray - direct paths**: From projector pixel to camera pixel via single scene point interaction.
* **Indirect paths**: Interaction with multiple scene points:
    * **Red rays - diffuse indirect paths**: Diffusely scatter at two or more diffuse points.
    * **Green rays - sepcular indirect paths**: Diffusely scatter at most once.

+++

Important: Since the area of a camera pixel is not infinitely small, it usually receives both, rays from direct and indirect light paths.

+++

The image captured with the camera can be imagined as a superposition of multiple latent images, all corresponding to different kinds of light paths.

+++

The reconstruction of these latent images is one major goal of light transport analysis.

+++

They often reveal interesting properties of the scene.

+++

### Teaser

We will see, how we

* can synthetically relight a scene, i.e., synthesize the image that would result when illuminating it with arbitrary illumination patterns (see previous example images),

+++

* can manipulate the image of a projector in order to compensate disturbing structures on the screen surface,

+++

* can separate direct an indirect light transport to obtain images dominated only by the respective optical effects.

<img src="figures/4/Example_Separation_Direct_Indirect.svg" style="max-width:30vw">

+++

### Entries of the light transport matrix

Each entry of $\mathbf{T}[m,n]$ describes the ratio of radiant energy that is transmitted from light source $n$ to sensor element $m$ via all optical paths that somehow connect source $n$ and sensor element $m$:
$\begin{align} 
   T[m,n] = \int\limits_{\Omega_{m,n}} f(x) \mathrm{d} \mu (x)\,,
\end{align}$
with $\Omega_{m,n}$ denoting the support, i.e., the space of light paths between source $n$ and sensor element $m$, the scattering throughput function $f(x)$ encoding the radiant energy transported along a light path $x$ and $\mu (x)$ representing the corresponding measure.

+++

#### Helmholtz reciprocity principle

For many practically relevant scenes, the scattering throughput scene can be assumed to obey the *Helmholtz reciprocity principle*, according to which the value of $f(x)$ for a light path $x$ does not depend on the propagation direction of the light rays of $x$ (i.e., interchanging projector pixel $n$ with camera pixel $m$ would result in the same value for $T[m,n]$).

+++

## Light transport analysis via optical computing

+++

When illuminating the scene with an illumination vector $\mathbf{p}$, capturing a camera image yields
$\begin{align} 
   \mathbf{i} = \mathbf{T}\mathbf{p} \,,
\end{align}$
i.e., the matrix multiplication $\mathbf{Tp}$ has been performed in the optical domain.

+++

This is the only possible way to interact with $\mathbf{T}$ in order to get information about it, i.e., via multiplying $\mathbf{T}$ with some vector $\mathbf{p}$ and observing the result $\mathbf{i}$.

+++

Luckily, dealing with very large matrices like $\mathbf{T}$ only via matrix multiplication as stated before, represents a well-studied problem in the field of linear algebra.

+++

Especially the set of so-called *Krylov subspace methods* is suitable for our case, which is why we will study some of these approaches in the following. 

### Optical power iteration

We will first study a simple method to obtain the eigenvector $\mathbf{v}_1$ of $\mathbf{T}$ that corresponds to its largest eigenvalue $\lambda_1$. For this purpose, we will transform a method called *power iteration* into the optical domain.

+++

The vector $\mathbf{v}$ is called the eigenvector of a square matrix $\mathbf{T}$ if 
$\begin{align} 
   \mathbf{Tv} = \lambda \mathbf{v} \,,
\end{align}$
with the so-called scalar eigenvalue $\lambda$.

+++

The eigenvector $\mathbf{v}_1$ that corresponds to the eigenvalue $\lambda_1$ with the highest absolute value is called the *principal eigenvector*.

+++

#### Power iteration

Power iteration is a simple numerical algorithm for calculating the principal eigenvector of a square matrix $\mathbf{T}$. It exploits the fact the sequence 
$\begin{align} 
   \mathbf{p}, \mathbf{Tp}, \mathbf{T}^2\mathbf{p}, \mathbf{T}^3\mathbf{p}, \ldots 
\end{align}$
converges to the principal eigenvector $\mathbf{v}_1$ of $\mathbf{T}$ for any initial vector $\mathbf{p}$, that is not orthogonal to $\mathbf{v}_1$.

+++

#### Implementation

+++ {"cell_style": "split"}

The typical implementation looks like:

$\mathbf{Function}\, \mathrm{powerIteration}(\mathbf{T})$<br><br>
$\quad \mathbf{p}_1 \leftarrow \text{ random vector}$<br>
$\quad \mathbf{for}\, k \in [1,\ldots,K]:$<br><br>
$\qquad \mathbf{i}_k \leftarrow \mathbf{Tp}_k$<br>
$\qquad \mathbf{p}_{k+1} \leftarrow \frac{\mathbf{i}_k}{\left\| \mathbf{i}_k \right\|_2 }$<br><br>
$\quad \textbf{return}\,  \mathbf{p}_{k+1}$

+++ {"cell_style": "split"}

The optical counterpart is given by:

$\mathbf{Function}\, \mathrm{opticalPowerIteration}(\mathtt{camera},\, \mathtt{projector})$<br><br>
$\quad \mathbf{p}_1 \leftarrow \text{ random vector}$<br>
$\quad \mathbf{for}\, k \in [1,\ldots,K]:$<br>
$\qquad \mathtt{projector}.\text{project}(\mathbf{p}_k)$<br>
$\qquad \mathbf{i}_k \leftarrow \mathtt{camera}.\text{acquireImage}()$<br>
$\qquad \mathbf{p}_{k+1} \leftarrow \frac{\mathbf{i}_k}{\left\| \mathbf{i}_k \right\|_2 }$<br><br>
$\quad \textbf{return}\,  \mathbf{p}_{k+1}$

+++

Note that all elements of $\mathbf{p}$ have to be non-negative so that they can be projected onto the scene.

+++

Some intermediate steps of an example optical power iteration application:

<img src="figures/4/powerIteration_example.svg" style="max-width:30vw">

+++

#### Limitations

+++

* Power iteration can only be applied to square matrices (i.e., for same sizes of projected images and captured images).

+++

* The convergence behavior of power iteration strongly depends on properties of $\mathbf{T}$, especially on the similarity of the top two eigenvalues.

+++ {"slideshow": {"slide_type": "slide"}}

### Krylov subspace methods

Krylov subspace methods represent a powerful family of algorithms for analyzing large sparse matrices $\mathbf{T}$ whose shape can be square or rectangular. All they need is to perform multiplications with $\mathbf{T}$ and sometimes with its transpose $\mathbf{T}\transp$.
<br>
Before we introduce the actual algorithms, we cover some prerequisites.

+++ {"slideshow": {"slide_type": "subslide"}}

The Krylov subspace of dimension $k$ is the span of vectors produced after $k$ steps via power iteration, i.e.:
$\begin{align} 
   \left( \mathbf{p}_1 , \mathbf{p}_2 = \mathbf{Tp}, \mathbf{p}_3 = \mathbf{T}^2 \mathbf{p}_1 , \ldots , \mathbf{p}_{k+1} = \mathbf{T}^k \mathbf{p}_1 \right) \,.
\end{align}$

+++ {"slideshow": {"slide_type": "subslide"}}

#### Optical matrix-vector multiplication for arbitrary vectors 

Krylov subspace methods require to compute products with $\mathbf{T}$ for arbitrary vectors $\mathbf{p}$, which may also contain negative values.

+++ {"slideshow": {"slide_type": "fragment"}}

For this purpose, we express $\mathbf{p}$ as the difference between two non-negative vectors $\mathbf{p}^+$ and $\mathbf{p}^-$:
$\begin{align} 
   \mathbf{p} &= \mathbf{p}^+ - \mathbf{p}^- \\
   \mathbf{Tp} &= \mathbf{Tp}^+ - \mathbf{Tp}^- \,.
\end{align}$

+++ {"slideshow": {"slide_type": "fragment"}}

As a result, we have to project and capture two images to achieve an optical realization of these kinds of matrix-vector multiplications.

+++

#### Symmetry of transport matrix

Only for symmetric matrices the convergence characteristics of Krylov subspace is well-understood and promises fast convergence. Hence, we stick to that case and ensure optically that the light transport matrix $\mathbf{T}$ is symmetric.

+++

##### Enforcing symmetry for $\mathbf{T}$

By using a coaxial camera-projector arrangement and equal resolutions for both components, the symmetry of $\mathbf{T}$ can be guaranteed due to Helmholtz reciprocity (see the following figure).

```{code-cell} ipython3
interact(lambda i: showFig('figures/4/symmetric_transport_matrix_',i,'.svg',800,50), i=widgets.IntSlider(min=(min_i:=1),max=(max_i:=7), step=1, value=(max_i if book else min_i)))
```

#### Low-rank approximation of $\mathbf{T}$ via optical Arnoldi

In its $k$-th iteration, the Arnoldi method allows to calculate the top $k$ eigenvectors (or singular vectors) of a matrix.

+++

It calculates a sequence of orthogonal vectors $\mathbf{p}_1, \ldots, \mathbf{p}_k$ whose span is an approximation of the span of the top $k$ eigenvectors of the matrix under investigation.

+++

The accuracy of the approximation increases with $k$.

+++

$\mathbf{Function}\, \mathrm{opticalSymmetricArnoldi}(\mathtt{camera},\, \mathtt{projector})$<br><br>
$\quad \mathbf{p}_1 \leftarrow \text{ non-zero random vector}$<br>
$\quad \mathbf{for}\, k \in [1,\ldots,K]:$<br>
$\qquad \mathtt{projector}.\text{project}(\mathbf{p}^+_k)$<br>
$\qquad \mathbf{i}^+_k \leftarrow \mathtt{camera}.\text{acquireImage}()$<br>
$\qquad \mathtt{projector}.\text{project}(\mathbf{p}^-_k)$<br>
$\qquad \mathbf{i}^-_k \leftarrow \mathtt{camera}.\text{acquireImage}()$<br>
$\qquad \mathbf{i}_k \leftarrow i^+_k - i^-_k$<br>
$\qquad \mathbf{p}_{k+1} \leftarrow \text{ortho}(\mathbf{p}_1, \ldots ,\mathbf{p}_k,\mathbf{i}_k)$
$\qquad \mathbf{p}_{k+1} \leftarrow \frac{\mathbf{p}_{k+1}}{\left\| \mathbf{p}_{k+1} \right\|_2 }$<br><br>
$\quad \textbf{return}\,  \left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]\transp  $

+++

The function $\text{ortho}(\mathbf{p}_1, \ldots ,\mathbf{p}_k,\mathbf{i}_k)$ projects its last argument $\mathbf{i}_k$ onto the subspace orthogonal to the columns of the matrix $\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_k]$, i.e.,
$\begin{align} 
  \text{ortho}(\mathbf{p}_1, \ldots ,\mathbf{p}_k,\mathbf{i}_k) = \mathbf{i}_k - \mathbf{P}\left( \mathbf{P}\transp \mathbf{P} \right)^{-1}  \mathbf{P}\transp \mathbf{i}_k
\end{align}$

+++

The Arnoldi algorithm constructs an orthogonal basis $\left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]$ for the subspace of illumination vectors and a basis $\left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right]$ for the subspace of acquired images. The product $\left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]\transp$ is a low-rank approximation of $\mathbf{T}$.

+++

In order to synthetically relight the scene with an arbitrary illumination $\mathbf{p}$, the following expression has to be evaluated:
$\begin{align} 
   \mathbf{Tp} \approx \left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]\transp \mathbf{p} \,.
\end{align}$

+++ {"tags": ["book_only"]}

```{note}
   If this equation is successively evaluated from right to left, at no time a matrix larger than $K \times N$ has to be kept in memory.
```

+++

#### Optical matrix inversion via GMRES

+++

We now assume that we are given a camera image $\mathbf{i}$ of our scene and want to obtain the illumination $\mathbf{p}$ that produced it, i.e., we want to find
$\begin{align} 
   \mathbf{p} = \argmin{\mathbf{x}} \left\| \left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]\transp \mathbf{x} - \mathbf{i} \right\|_2 \,. 
\end{align}$

+++

This optimization problem can be solved via the Moore-Penrose pseudoinverse $\mathbf{T}^\dagger$ of $\mathbf{T}$ (in order to account for a potentially singular matrix $\mathbf{T}$):
$\begin{align} 
   \mathbf{p} = \mathbf{T}^\dagger \mathbf{i}\,.
\end{align}$

+++

This solution can also be calculated optically via another popular Krylov subspace method called *generalized minimum residual (GMRES)* that also only relies on computing products with $\mathbf{T}$ and does not need to access to the complete matrix.

+++

The algorithm differs only slightly from optical Arnoldi, namely
* in the initialization vector $\mathbf{p}_1$, which is always the target image $\mathbf{i}$ and
* in the return value, which is $\left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]^\dagger \mathbf{i}$, i.e., the approximation of the sought illumination vector.

+++

$\mathbf{Function}\, \mathrm{opticalSymmetricGMRES}(\mathtt{camera},\, \mathtt{projector}, \mathbf{i})$<br><br>
$\quad \mathbf{p}_1 \leftarrow \mathbf{i}$<br>
$\quad \mathbf{for}\, k \in [1,\ldots,K]:$<br>
$\qquad \mathtt{projector}.\text{project}(\mathbf{p}^+_k)$<br>
$\qquad \mathbf{i}^+_k \leftarrow \mathtt{camera}.\text{acquireImage}()$<br>
$\qquad \mathtt{projector}.\text{project}(\mathbf{p}^-_k)$<br>
$\qquad \mathbf{i}^-_k \leftarrow \mathtt{camera}.\text{acquireImage}()$<br>
$\qquad \mathbf{i}_k \leftarrow i^+_k - i^-_k$<br>
$\qquad \mathbf{p}_{k+1} \leftarrow \text{ortho}(\mathbf{p}_1, \ldots ,\mathbf{p}_k,\mathbf{i}_k)$
$\qquad \mathbf{p}_{k+1} \leftarrow \frac{\mathbf{p}_{k+1}}{\left\| \mathbf{p}_{k+1} \right\|_2 }$<br><br>
$\quad \textbf{return}\,  \left[ \mathbf{i}_1 \cdots \mathbf{i}_{K} \right] \left[ \mathbf{p}_1 \cdots \mathbf{p}_K  \right]^\dagger \mathbf{i}  $

+++

Intuitively, GMRES calculates a rank-$K$ approximation of $\mathbf{T}$ and calculates its pseudoinverse to solve for $\mathbf{p}$. The initialization of $\mathbf{p}_1$ with $\mathbf{i}$ helps to explore only a portion of $\mathbf{T}$'s row space that is suitable for the inversion with respect to $\mathbf{i}$.

+++

##### Example experiments

<img src="figures/4/result_GMRES.svg" style="max-width:30vw">

+++

## Primal-Dual coding for optical probing

+++

*Optical probing* can be seen as the optical implementation of *matrix probing*, a topic of numerical mathematics dealing with the efficient estimation of special regions of interest of very large matrices (like $\mathbf{T}$), e.g., its trace or diagonal. 

+++

Optical probing allows to obtain certain latent images of the scene which are normally hidden in the global light transport. 

+++

For example, it is possible to obtain images corresponding to either (approximately) only direct or (approximately) only indirect light transport. Such images can reveal interesting insights about the observed scene (e.g., the structure of the blood vessels under the skin). 

+++

Optical probing can be mathematically modelled as
$\begin{align} 
   \mathbf{i} = \left( \boldsymbol{\Pi} \odot \mathbf{T} \right) \mathbf{1} \,,
\end{align}$
with $\boldsymbol{\Pi}$ denoting the *probing matrix* and $\odot$ denoting the element-wise product between two matrices of equal size, $\mathbf{1}$ denoting a vector of all ones representing a uniform illumination and finally $\mathbf{i}$ representing the image captured under uniform illumination and for a light transport matrix that is the result of $\boldsymbol{\Pi} \odot \mathbf{T}$.

+++

In order to be able to manipulate $\mathbf{T}$ in this way, it is necessary to control two aspects of the image formation process: 
* the illumination, the so-called *primal domain* via the projected image and
* the pixel-wise modulation of the camera's sensor, the so-called *dual domain*.

+++

Control of the dual domain is possible by 



```{code-cell} ipython3

```

```{code-cell} ipython3
#Generate synth. LTM
SIZE = 128
RANK = 90
LTM = np.random.randn(SIZE,RANK)@np.random.randn(RANK,SIZE)
LTM = LTM + 1e-3 * np.random.randn(SIZE)
```

```{code-cell} ipython3
NUM_ITERATIONS = 128#1024     # Number of Arnoldi iterations
NUM_ITERATIONS_STEP = 1   # Only reconstruct matrix every 
```

```{code-cell} ipython3
import h5py
f = h5py.File('Green.mat','r')
data = f.get('LTM')
data = np.array(data) # For converting to a NumPy array
LTM = data
NUM_ITERATIONS = 1024     # Number of Arnoldi iterations
NUM_ITERATIONS_STEP = 32   # Only reconstruct matrix every 
```

```{code-cell} ipython3
M,N = LTM.shape
```

```{code-cell} ipython3
right_Arnoldi_vectors = np.zeros((N,NUM_ITERATIONS + 1))
right_Arnoldi_vectors[:,0] = (a:=np.ones(N))/np.linalg.norm(a)
left_Arnoldi_vectors  = np.zeros((M,NUM_ITERATIONS))
```

```{code-cell} ipython3
for k in range(0,NUM_ITERATIONS):
    
    tmp = LTM@right_Arnoldi_vectors[:,k];    
    left_Arnoldi_vectors[:,k] = tmp;
    
    tmp = LTM.transpose()@left_Arnoldi_vectors[:,k];
    inv = np.linalg.inv(right_Arnoldi_vectors[:,0:k+1].transpose()@right_Arnoldi_vectors[:,0:k+1])
    print(inv.shape)
    right = right_Arnoldi_vectors[:,0:k+1].transpose()@tmp
    right = inv @ right
    right = right_Arnoldi_vectors[:,0:k+1] @ right
    tmp = tmp - right
    #tmp = tmp - right_Arnoldi_vectors[:,0:k+1] @ inv @right_Arnoldi_vectors[:,0:k+1].transpose()@tmp
    tmp = tmp/np.linalg.norm(tmp);

    right_Arnoldi_vectors[:,k+1] = tmp;
```

```{code-cell} ipython3
a,b = optArnoldi(NUM_ITERATIONS, LTM)
```

```{code-cell} ipython3
errors = []
for k in range(0, NUM_ITERATIONS, NUM_ITERATIONS_STEP):
    LTM_approx = left_Arnoldi_vectors[0,0:k+1]@right_Arnoldi_vectors[0,0:k+1].transpose()
    #errors.append(np.linalg.norm(LTM - LTM_approx) / np.linalg.norm(LTM))
    errors.append(np.sum(np.abs(LTM-LTM_approx)))
```

```{code-cell} ipython3
errors2 = []
for k in range(0, NUM_ITERATIONS, NUM_ITERATIONS_STEP):
    LTM_approx = a[0,0:k+1]@b[0,0:k+1].transpose()
    #LTM_approx = b[0,0:k+1]@a[0,0:k+1].transpose()
    #errors2.append(np.linalg.norm(LTM - LTM_approx) / np.linalg.norm(LTM))
    errors2.append(np.sum(np.abs(LTM-LTM_approx)))
```

```{code-cell} ipython3
tmp = LTM.transpose()@LTM
U,S,right_singular_vectors = np.linalg.svd(tmp)

left_singular_vectors = LTM@right_singular_vectors

errors3 = []
for k in range(0, NUM_ITERATIONS, NUM_ITERATIONS_STEP):
    LTM_approx = left_singular_vectors[:,0:k+1]@right_singular_vectors[:,0:k+1].transpose()
    #errors2.append(np.linalg.norm(LTM - LTM_approx) / np.linalg.norm(LTM))
    errors3.append(np.sum(np.abs(LTM-LTM_approx)))
```

```{code-cell} ipython3
plt.figure()
plt.plot(errors)
```

```{code-cell} ipython3
plt.figure()
plt.plot(errors2)
```

```{code-cell} ipython3
plt.figure()
plt.plot(errors3)
```

```{code-cell} ipython3
def optArnoldi(K, A:np.ndarray):
    i = np.zeros((A.shape[0], K))
    p = np.zeros((A.shape[0], K+1))
    #p[:,0] = np.random.rand(A.shape[1])
    p[:,0] = (a:=np.ones(A.shape[1]))/np.linalg.norm(a)

    for k in range(0,K):
        i[:,k] = A@p[:,k]
        P = p[:,0:k+1]
        #print(P.shape)
        #print(P)
        p[:,k+1] = i[:,k] - P@np.linalg.inv(P.transpose() @ P)@P.transpose()@i[:,k]
        p[:,k+1] = p[:,k+1] / np.linalg.norm(p[:,k+1])
        
    return i, p[:,:-1]
    
```

```{code-cell} ipython3
def classicArnoldi(K, A:np.ndarray):
    eps = 1e-10
    v1 = (a:=np.ones(A.shape[1]))/np.linalg.norm(a)
    H = np.zeros((K+1,K+1))
    W = np.zeros((A.shape[0],K+1))
    V = np.zeros((A.shape[0],K+1))
    V[:,0] = v1
    for j in range(0,K):
        for i in range(0, j+1):
            H[i,j] = np.dot(A@V[:,j], V[:,i])
        tmp = np.zeros((A.shape[0]))
        for i in range(0,j+1):
            tmp = tmp + H[i,j]*V[:,i]
        print(tmp.shape)
        moep = A@V[:,j]
        print(moep.shape)
        W[:,j] = A@V[:,j] - tmp
        H[j+1, j] = np.linalg.norm(W[:,j])
        print(f"{j}: {H[j+1,j]}")
        if (H[j+1,j] < eps):
            print("moep")
            break
        V[:,j+1] = W[:,j] / H[j+1,j]
    return H, W, V
```

```{code-cell} ipython3
K2 = 128
h,w,v = classicArnoldi(K2, LTM)
```

```{code-cell} ipython3
v = v[:,0:-1]
h = h[0:-1,0:-1]
```

```{code-cell} ipython3
plt.figure()
plt.imshow(h)
```

```{code-cell} ipython3
LTM_approx = v@h@v.transpose()
```

```{code-cell} ipython3
errors4 = []
for k in range(0, K2):
    LTM_approx = v[:,0:k+1]@h[0:k+1,0:k+1]@v[:,0:k+1].transpose()
    errors4.append(np.sum(np.abs(LTM-LTM_approx)))
```

```{code-cell} ipython3
plt.figure()
plt.plot(errors4)
plt.plot(errors3)
```

```{code-cell} ipython3
plt.figure()
plt.imshow(np.abs(LTM-LTM_approx))
```
