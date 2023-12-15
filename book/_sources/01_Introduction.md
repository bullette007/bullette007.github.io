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
                  if (tags[j]=="book_only" | tags[j]=="remove-cell") {cur_cell.element.hide()};
                  if (tags[j]=="presentation_only") {cur_cell.element.show();}
            }}}
```

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

+++ {"slideshow": {"slide_type": "slide"}}

# Introduction

+++ {"slideshow": {"slide_type": "fragment"}}

##### Content
* What is computational imaging
* Motivating examples
* Course roadmap / tentative schedule
* Organizational information

+++ {"slideshow": {"slide_type": "subslide"}}

## What is computational imaging

+++ {"tags": ["book_only"]}

Imaging is the process of acquiring an image of a scene of interest that containes details of interest of the scene.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Example: Conventional optical imaging

+++ {"tags": ["book_only"]}

Conventional optical imaging (e.g., with a digital camera) usually involves a high-quality optical system consisting of multiple lenses (around 7 in 2020 high-end smartphones), a high-resolution sensor chip (around 12 megapixel in smartphones, up to 40 megapixel in digital single-lens reflex cameras) and some image processing algorithms to improve image quality with respect to the application (photography, microscopy, medical imaging, etc.).

+++ {"slideshow": {"slide_type": "fragment"}}

<img src="figures/1/conventional_imaging.svg" style="max-height:40vh">

+++ {"tags": ["book_only"]}

Each of the three components is usually optimized for its own, i.e.:
* the optics are designed to produce a sharp image of the scene on the sensor plane,
* the sensor is optimized to digitize the incident light intensities as fast as possible with a minimum of noise and
* the image processing algorithms are developed to mitigate general forms of remaining noise or visual artifacts.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Computational imaging

+++ {"tags": ["book_only"]}

In computational imaging the individual parts (i.e., optics, sensor, algorithms) are holistically considered as a single system and are optimized together to yield the best possible results with respect to the application (examples will be shown later).

+++ {"tags": ["presentation_only", "remove-cell"]}

Holistic, joint optimization of all system components:

+++

<img src="figures/1/computational_imaging.svg" style="max-height:40vh">

+++ {"tags": ["book_only"]}

As a main consequence of this approach, the intermediate image that is formed on the sensor may appear blurred or otherwise scrumbled. However, since the details of the image formation process are known, they can be mitigated by the image processing algorithms and the sought-after image can be recovered.

+++ {"slideshow": {"slide_type": "slide"}}

## Degrees of freedom / design space

Various parameters of the involved components can be varied to achieve optimal system performance.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Illumination

+++ {"cell_style": "split"}

<img src="figures/1/illumination.svg" style="max-height:40vh">

+++ {"cell_style": "split"}

The illumination can be varied in terms of
* intensity,
* emission direction,
* emission profile,
* spectrum,
* polarization,
* spatial pattern, etc.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Image acquisition

+++ {"cell_style": "split"}

<img src="figures/1/sensor.svg" style="max-height:40vh">

+++ {"cell_style": "split"}

The image acquisition can be varied in terms of
* lens / lens system,
* focal length,
* shutter,
* (programmable) aperture,
* filters,
* integration time etc.

+++ {"tags": ["presentation_only", "remove-cell"], "slideshow": {"slide_type": "subslide"}}

$\rightarrow $ Exploit all these parameters to achieve optimal performance with respect to the imaging task on hand.

+++ {"tags": ["book_only"]}

To finally obtain an image which is optimal for the application on hand, all the variable parameters have to be computationally controlled and the intermediate image(s) have to be adequately processed. In other words, the image of interest has to be **reconstructed**.

Hence, in most computational imaging algorithms, the crucial part is represented by an image reconstruction procedure that tries to solve a so-called **inverse problem**. Usually, the forward process, i.e., the formation of an image on the sensor plane for a given scene, can be (physically) modelled and computed. The task of the algorithms is inversion of that process, i.e., to reconstruct the most probable scene which led to the observed image on the sensor.

+++ {"slideshow": {"slide_type": "subslide"}}

Computational imaging allows to 
* acquire more information about the scene of interest as with conventional imaging and
* to reduce the weight, cost and complexity of lens systems by shifting parts of the image formation process into the digital domain.

+++ {"slideshow": {"slide_type": "slide"}}

## Motivating Examples

+++ {"slideshow": {"slide_type": "subslide"}}

### Photography

+++ {"tags": ["book_only"]}

```{note}
Computational imaging applied to photography applications is often referred to as *computational photography*.   
```

+++ {"slideshow": {"slide_type": "subslide"}}

**Reduction of weight and cost**

+++ {"tags": ["book_only"], "slideshow": {"slide_type": "-"}}

<br> By means of computational imaging, the number and required quality of the lenses of a camera can be reduced without negatively impacting the quality of the resulting images. The intermediate image on the sensor may appear unsatisfying to the naked eye (e.g., blurred) but the sought-after sharp image is reconstructed by the image processing algorithms. This reduces the cost, weight and complexity of the optical system of the camera.<br> In the extreme case, it is even possible to obtain sharp images without employing a lens at all (so-called *lensless imaging*).

+++ {"cell_style": "center", "slideshow": {"slide_type": "fragment"}}

**Diffuser Cam**<br> 
(Grace Kuo, Nick Antipa, Ren Ng, and Laura Waller. "DiffuserCam: Diffuser-Based Lensless Cameras." Computational Optical Sensing and Imaging. Optical Society of America, 2017)

+++ {"slideshow": {"slide_type": "subslide"}}

**Lensless camera**<br>
(Xiuxi Pan et al, Image reconstruction with transformer for mask-based lensless imaging, Optics Letters (2022). DOI: 10.1364/OL.455378)

+++ {"slideshow": {"slide_type": "subslide"}}

**High dynamic range (HDR) imaging**

+++ {"tags": ["book_only"]}

Most commonly, images captured with digital cameras are stored with 8 bit precision per color channel (more details later). As a consequence, images of scenes with high dynamics regarding the light intensities will lose information resulting in either white oversaturated or completely black areas. <br> HDR imaging methods acquire a series of images with varying integration times and processes the images either to obtain a high dynamic range image (with more than 8 bit per channel) or to compress the dynamic range so that no images regions appear oversaturated or under exposed (so-called **tone mapping**).

+++ {"slideshow": {"slide_type": "subslide"}}

**Single-shot high dynamic range imaging** <br>
(Metzler, C., Ikoma, H., Peng, Y., Wetzstein, G., Deep Optics for Single-shot High-dynamic-range Imaging, CVPR 2020)

+++ {"slideshow": {"slide_type": "subslide"}}

**Light field imaging**

+++ {"tags": ["book_only"]}

In contrast to conventional cameras, light field cameras also capture the direction of incidence with which light rays enter the camera. This additional information allows to post-process the acquired to image to change the viewing perspective, to alter the focus distance or to enhance the depth of field.

+++ {"slideshow": {"slide_type": "subslide"}}

**Post-acquisition refocusing**

+++ {"cell_style": "split"}

<img src="figures/3/lfDefocused.jpg" style="max-height:40vh">

+++ {"cell_style": "split"}

<img src="figures/3/lfFocused.jpg" style="max-height:40vh">

+++ {"slideshow": {"slide_type": "subslide"}}

**Coded aperture imaging**

+++ {"tags": ["book_only"]}

In an optical system, the aperture is the optical or mechanical element that mostly limits the amount of light reaching the sensor (more details later). Making this element computationally controllable provides another handle for influencing the image formation process. By this means it is possible, e.g., to deblur images which are affected by motion blur or which have been acquired out of focus or to additionally capture depth information.

+++ {"slideshow": {"slide_type": "subslide"}}

**Coded aperture imaging for image and depth acquisition**<br>
(Levin, A., Fergus, R., Durand, F.,  Freeman, W. T. (2007). Image and depth from a conventional camera with a coded aperture. ACM transactions on graphics (TOG), 26(3), 70-es.)

+++ {"slideshow": {"slide_type": "subslide"}}

### Medical imaging and microscopy

+++ {"slideshow": {"slide_type": "subslide"}}

**Tomography**

+++ {"tags": ["book_only"]}

In many medical imaging applications one is interested in obtaining threedimensionally resolved images about the internal structures of the human body without having to physically interacting with it. By means of tomographic reconstruction algorithms, several projection images acquired with penetrating radiation (e.g., X-rays) can be combined and processed to obtain an image slice at the position of interest.

+++ {"slideshow": {"slide_type": "subslide"}}

**Coded illumination for microscopic phase imaging**<br>
(Kellman, M. R., Bostan, E., Repina, N. A., Waller, L. (2019). Physics-based learned design: optimized coded-illumination for quantitative phase imaging. IEEE Transactions on Computational Imaging, 5(3), 344-353.)

+++ {"tags": ["book_only"]}

In medical microscopy one is often interested in the phase (i.e., the direction of propagation) of the light that has been transmitted through a sample, rather than in its intensity (conventional imaging). This can be achieved by adequately processing a series of images acquired under varying illumination patterns learned with machine learning approaches.

+++ {"slideshow": {"slide_type": "subslide"}}

**Fourier Ptychography**

+++ {"tags": ["book_only"]}

In microscopy, the achievable lateral resolution (i.e., the resolutions along the axes parpendicular to the optical axis) is linked to the numerical aperture (more later) of the lens system. By acquiring multiple images under varying illumination one can reconstruct the complex phase via Fourier Ptychography resulting in an increased synthetic numerical aperture. By this means it is possible to up to double the resolution.

+++ {"slideshow": {"slide_type": "subslide"}}

### Visual inspection

+++ {"slideshow": {"slide_type": "subslide"}}

**Deflection map acquisition and processing**

+++ {"tags": ["book_only"]}

The visual inspection of transparent objects is especially challenging. This is because the material defects of interest (scratches, inclusions of air bubbles, etc.) are transparent themselves and hence do not affect the light's intensity. Instead, those defects change the propagation direction of light that has been transmitted through the test object. By acquiring the distribution of the light's propagation direction behind the test object with high resolution (so-called deflection maps) and by processing them with suitable algorithms, material defects can be visualized with high contrast.

+++ {"slideshow": {"slide_type": "fragment"}}

<img src="figures/1/LightfieldLaserScannerWithObject_3.svg" style="max-height:40vh">

+++ {"cell_style": "split", "slideshow": {"slide_type": "subslide"}}

<img src="figures/1/kasPhoto.svg" style="max-height:40vh">

+++ {"cell_style": "split"}

<img src="figures/1/kasExperiments.svg" style="max-height:40vh">

+++ {"slideshow": {"slide_type": "subslide"}}

**Inverse (light field) illumination**

+++ {"tags": ["book_only"]}

Both opaque and transparent test objects can be illuminated with specifically adapted, so-called inverse light fields that render their intended structures invisible to the camera but reveals material defects with high contrast. Such approaches allow to inspect the whole test object by acquiring a single image only.

+++ {"slideshow": {"slide_type": "fragment"}, "cell_style": "split"}

<img src="figures/1/invLF_acquisition_2.svg" style="max-height:40vh">

+++ {"slideshow": {"slide_type": "fragment"}, "cell_style": "split"}

<img src="figures/1/invLF_emission_4.svg" style="max-height:40vh">

+++ {"slideshow": {"slide_type": "subslide"}, "cell_style": "center"}

<img src="figures/1/invLF_simu_exp.svg" style="max-height:90vh">

+++ {"slideshow": {"slide_type": "slide"}}

## Course roadmap / tentative schedule

+++ {"slideshow": {"slide_type": "fragment"}}

Tentative course schedule:

| Nr. | Date       | Lecture (14:00 - 15:30)                                                              | Exercise (15:45 - 17:15)                           |
|-----|------------|--------------------------------------------------------------------------------------|----------------------------------------------------|
| 1   | 25.10.2023 | Introduction  Basics                                                                 |                                                    |
| 2   | 01.11.2023 | No lecture - public holiday                                                          |                                                    |
| 3   | 08.11.2023 | Basics                                                                               |                                                    |
| 4   | 15.11.2023 | Light field methods                                                                  | Fourier transforms  Mitsuba renderer               |
| 5   | 22.11.2023 | Light field methods                                                                  |                                                    |
| 6   | 29.11.2023 | Light field methods                                                                  | Light field calculations  Mitsuba for light fields |
| 7   | 06.12.2023 | Light transport analysis                                                             |                                                    |
| 8   | 13.12.2023 | Neural networks for computational imaging                                            |                                                    |
| 9   | 20.12.2023 | Neural networks for computational imaging  Inverse problems in computational imaging | Light transport calculations                       |
| 10  | 10.01.2024 | Inverse problems in computational imaging                                            |                                                    |
| 11  | 17.01.2024 | Inverse problems in computational imaging                                            | Inverse problems                                   |
| 12  | 24.01.2024 | Lensless imaging                                                                     |                                                    |
| 13  | 31.01.2024 | Coded exposure photography   Quantitative phase imaging                              | Inverse problems                                   |
| 14  | 07.02.2024 | Coded spectral snapshot imaging                                                      |                                                    |
| 15  | 14.02.2024 | Time of flight imaging  Round up                                                     | DiffuserCam                                        |

+++ {"tags": ["remove-cell"]}

The following topics will be covered in the remaining classes of this course:
* Introduction
    * What is computational photography?
    * Motivating example applications (photography, medical imaging, visual inspection)
    * Course roadmap
* Fundamental basics
    * Optics
    * Illumination
    * Image acquisition and image formation process (including basic system theory)
    * Digital image representation
* Light field methods
    * Introduction into light fields
    * Photography applications
        * Light field camera
        * Digital refocusing
        * Visualizing refractive phenomena in transparent media
            * Schlieren imaging
            * Light field illumination (directional light field probes)
    * Visual inspection applications
    * Light field laser scanning
    * Light deflection map processing
    * Inverse illumination (work by Dr. Gruna)
    * Inverse light field illumination
* Light transport analysis
    * Optical power iterations
    * Light transport matrices
    * Indirect light imaging
* Holistic optical design
    * Deconvolution / inverse problems
    * Closed-form methods (inverse filter, Wiener filter)
    * Iterative methods
        * ADMM (alternating direction method of multipliers)
    * Compressed sensing
    * Coded aperture imaging
        * Transmission masks
        * Phase masks
        * Hadamard codes
        * Lensless imaging
•	Further topics
    * Phase imaging
    * Fourier Ptychography

+++ {"slideshow": {"slide_type": "slide"}}

## Organizational information

+++ {"slideshow": {"slide_type": "subslide"}}

##### Course times

* Every wednesday from 14:00 - 15:30: primary lecture content.
* Additionally every other wednesday from 15:45 - 17:15: discussion of exercises.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Christmas

Last lecture before christmas brake on 20.12.2022.

+++ {"slideshow": {"slide_type": "subslide"}}

##### Exam

Can be oral or written (depends a bit on number of participants).<br>
Date for potential written exam: Monday 11.03.2024 11:00 - 12:00. NTI Hörsaal (Nachrichtentechnik)

+++ {"slideshow": {"slide_type": "subslide"}}

##### Course material

Slides (i.e., Jupyter notebooks), lecture notes and exercises (including solutions) will be made available on https://computational-imaging.de.

You can send me questions any time to johannes.meyer@iosb.fraunhofer.de.
