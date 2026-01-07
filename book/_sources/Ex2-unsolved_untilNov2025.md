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

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Light-Field-Methods" data-toc-modified-id="Light-Field-Methods-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Light Field Methods</a></span><ul class="toc-item"><li><span><a href="#Reading-and-displaying-light-fields" data-toc-modified-id="Reading-and-displaying-light-fields-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Reading and displaying light fields</a></span><ul class="toc-item"><li><span><a href="#Get-light-field-data" data-toc-modified-id="Get-light-field-data-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Get light field data</a></span></li><li><span><a href="#Reading-the-light-field" data-toc-modified-id="Reading-the-light-field-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Reading the light field</a></span></li><li><span><a href="#Displaying-the-center-sub-aperture-image-(SAI)" data-toc-modified-id="Displaying-the-center-sub-aperture-image-(SAI)-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Displaying the center sub aperture image (SAI)</a></span></li><li><span><a href="#Display-different-SAIs" data-toc-modified-id="Display-different-SAIs-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>Display different SAIs</a></span></li><li><span><a href="#Synthesize-a-full-aperture-image" data-toc-modified-id="Synthesize-a-full-aperture-image-2.1.5"><span class="toc-item-num">2.1.5&nbsp;&nbsp;</span>Synthesize a full aperture image</a></span></li></ul></li><li><span><a href="#Refocusing-with-light-fields" data-toc-modified-id="Refocusing-with-light-fields-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Refocusing with light fields</a></span></li><li><span><a href="#Focus-stack" data-toc-modified-id="Focus-stack-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Focus stack</a></span></li><li><span><a href="#Mitsuba-plugin-for-light-field-camera" data-toc-modified-id="Mitsuba-plugin-for-light-field-camera-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Mitsuba plugin for light field camera</a></span></li><li><span><a href="#Schlieren-imaging" data-toc-modified-id="Schlieren-imaging-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Schlieren imaging</a></span><ul class="toc-item"><li><span><a href="#Quantitative-measurement-of-deflection-angles" data-toc-modified-id="Quantitative-measurement-of-deflection-angles-2.5.1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>Quantitative measurement of deflection angles</a></span></li><li><span><a href="#Mitsuba-plugin-for-schlieren-sensor" data-toc-modified-id="Mitsuba-plugin-for-schlieren-sensor-2.5.2"><span class="toc-item-num">2.5.2&nbsp;&nbsp;</span>Mitsuba plugin for schlieren sensor</a></span></li></ul></li><li><span><a href="#Light-field-displays" data-toc-modified-id="Light-field-displays-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Light field displays</a></span></li></ul></li></ul></div>

```{code-cell} ipython3
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as sig
from scipy.integrate import simpson
from IPython.display import SVG, display, IFrame, HTML
#%matplotlib notebook
%matplotlib widget
from scipy import ndimage
def imshow(img, cmap=None):
    plt.close('all')
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()
```

# Light Field Methods

+++ {"slideshow": {"slide_type": "subslide"}}

## Reading and displaying light fields

+++

### Get light field data

+++

Download the HCI light field dataset (see 'A dataset and evaluation methodology for depth estimation on 4D light fields' by Honauer et al.) from https://lightfield-analysis.uni-konstanz.de/

+++

### Reading the light field

+++

Read light field from the bedroom scene from the HCI dataset and convert it into a 4D-structure `lf(m,n,j,k)` with `(m,n)` denoting the spatial coordinates and `(j,k)` denoting the angular coordinates.

+++

### Displaying the center sub aperture image (SAI)

+++ {"slideshow": {"slide_type": "subslide"}}

Show center SAI for $\mathbf{j}=(4,4)^\intercal$:

+++

### Display different SAIs

+++

Visualize how the perspective changes when the horizontal angular coordinate is changed continuosly.

+++

### Synthesize a full aperture image

+++ {"slideshow": {"slide_type": "subslide"}}

Synthesize full aperture image, i.e., $g(\mathbf{m})=\sum\limits_{\mathbf{j}\in\Omega_\mathrm{a}} L(\mathbf{m},\mathbf{j})\,$:

+++

## Refocusing with light fields

+++ {"slideshow": {"slide_type": "subslide"}}

Define a refocus function `refocus(alpha)` by means of the shift and add-method as shown in the lecture.

+++

Hint: To realize the shift-operation, have a look at OpenCV whether there is a suitable function.

+++ {"slideshow": {"slide_type": "subslide"}}

Visualize the refocusing results for different $\alpha$:

+++

## Focus stack

+++

We will now calculate a single all-in-focus image of the observed scene.<br>
For that purpose, first create a stack of refocused images for the $\alpha$-values used above. Then try to filter each image of this series with a filter, that highlights sharp image regions, i.e., image regions with high spatial frequencies. For every pixel position, find the image inside the stack that is most sharp at that position. Eventually, extract for every pixel position the pixel value of the focal stack image that exhibits the highest sharpness for that position in order to obtain the all-in-focus image.

+++

## Mitsuba plugin for light field camera

+++

Implement a Mitsuba plugin that realizes a light field camera based on a micro-lens array as introduced in the lecture. Assume circular lenses and no crosstalk. Save the light field as a single spatially multiplexed image.

```{code-cell} ipython3
variant = "scalar_rgb"
#variant = "cuda_ad_rgb"
#variant = "llvm_ad_rgb"

import os as os
os.environ["MI_DEFAULT_VARIANT"] = variant

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant(variant)
```

```{code-cell} ipython3
class light_field_sensor (mi.Sensor):
    def __init__(self, props):
        mi.Sensor.__init__(self, props)
        self.to_world = props['to_world']
        self.foc_len  = props['foc_len']
        self.img_dist_l = props['img_dist_l']
        self.img_dist_ml = props['img_dist_ml']
        self.sen_size = props['sen_size']
        
        self.ang_res = props['ang_res']
        self.filmsize = (self.sen_size, self.sen_size)
        
        self.spat_res = self.film().size()[0] / self.ang_res
        self.ml_diam = self.sen_size / self.spat_res
        self.ml_rad = self.ml_diam / 2
            
    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        
        # ...
        
        rayd = mi.RayDifferential3f()
        rayd.has_differentials = False
        
        #rayd.o = ...
        #rayd.d = ...
        
        
        d = dr.normalize(mi.Vector3f(...))
                
        rayd.o = self.to_world.transform_affine(...)
        rayd.d = self.to_world @ d
        
        
        
        return (rayd, mi.Color3f(weight, weight, weight))
```

```{code-cell} ipython3
mi.register_sensor("light_field_sensor", lambda props: light_field_sensor(props))
```

```{code-cell} ipython3
g = 8.0
b = 12
b_ml = 4
f = 1/(1/b + 1/g)
#f,b,g,V
```

```{code-cell} ipython3
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path'
    },

    'sphere' : {
        'type': 'sphere',
        'emitter': {
            'type': 'area',
            'radiance': {
            'type': 'rgb',
            'value': 0.8,
        },
    },
    'center': [0, 0, g],
    'radius': 0.7,
        
    },
    'sphere2' : {
        'type': 'sphere',
        'emitter': {
            'type': 'area',
            'radiance': {
            'type': 'rgb',
            'value': 0.8,
        },
    },
    'center': [-2, -2, g+1],
    'radius': 0.4,
        
    },
 
    'sensor': {
        'type': 'light_field_sensor',
        'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, 0],
                                                 target=[0, 0, 1],
                                                 up=[0, 1, 0]),
        'foc_len' : f,
        'img_dist_l': b,
        'img_dist_ml': b_ml,
        'sen_size': 10.0,
        'ang_res': 5,
        'film': {'type': 'hdrfilm',
      'width': 500,
      'height': 500,
      'rfilter': {'type': 'gaussian'},
      'pixel_format': 'rgb',
      'component_format': 'float32'},
     'sampler': {'type': 'independent', 'sample_count': 4},
    }
})
```

```{code-cell} ipython3
image = mi.render(scene)
```

```{code-cell} ipython3
plt.figure()
plt.imshow(image, cmap='gray')
```

```{code-cell} ipython3

```

## Schlieren imaging

+++

### Quantitative measurement of deflection angles

+++

In the lecture we learned:

+++

Quantitative measurements of the deflection angle $\alpha$ are only possible if

+++

$$ \delta_\alpha > \frac{1}{2}\varepsilon\,, $$

+++

with $\varepsilon$ denoting the diameter of the image of the light source.

+++

Explain why this is true.

+++

### Mitsuba plugin for schlieren sensor

+++

Implement a color-coded Schlieren setup in the Mitsuba rendering framework.

+++

Encode the color-wheel via the HSI color space as follows:
* The angle is encoded via the hue value and
* the radius is encoded via the intensity value.

```{code-cell} ipython3
class color_schlieren_sensor (mi.Sensor):
    def __init__(self, props):
        mi.Sensor.__init__(self, props)
        self.to_world = props['to_world']
        self.foc_len  = props['foc_len']
        self.img_dist = props['img_dist']
        self.sen_size = props['sen_size']
        self.lens_rad = props['lens_rad']
        self.filmsize = (self.sen_size, self.sen_size)
        self.rad_max = (self.lens_rad - self.sen_size / 2) / self.img_dist * (self.img_dist - self.foc_len) + self.sen_size / 2
        
    def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
        
        
        # TODO

    
    def to_string(self):
        return ('thin_lens_sensor[\n'
                '    foc_len=%s,\n'
                '    img_dist=%s,\n'
                '    sen_size=%s,\n'
                '    lens_rad=%s,\n'
                ']' % (self.foc_len, self.img_dist, self.sen_size, self.lens_rad))
        
```

```{code-cell} ipython3
mi.register_sensor("color_schlieren_sensor", lambda props: color_schlieren_sensor(props))
```

```{code-cell} ipython3
class telecentric_area_light (mi.Emitter):
    def __init__(self, props):
        mi.Emitter.__init__(self, props)
        self.acc_angle = dr.deg2rad(props['acc_angle'])
        self.radiance = mi.Color3f(props['radiance'])
        self.flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying
    
    def angleBetweenVectors(self, a:mi.Vector3f, b:mi.Vector3f):
        return dr.acos(dr.dot(a,b) / (dr.norm(a) * dr.norm(b)))
    
    def eval(self, si:mi.SurfaceInteraction3f, active=True):
        
        res = mi.Color3f(0.0, 0.0, 0.0)
        
        if self.angleBetweenVectors(si.n, si.wi * -1.0) <= self.acc_angle:
            res = self.radiance
        
        return res

    def sample_direction(self, it:mi.Interaction3f, sample, active=True):
        dirsam = mi.DirectionSample3f()
        return (mi.DirectionSample3f(), mi.Color3f())
    
    def to_string(self):
        return ('telecentric_area_light[\n'
                '    acc_angle=%s,\n'
                '    radiance=%s,\n'
                ']' % (dr.rad2deg(self.acc_angle), self.radiance))
mi.register_emitter("telecentric_area_light", lambda props: telecentric_area_light(props))
```

```{code-cell} ipython3
g = 8.0
V = 2.0
f = g/(1/V+1)
b = 1/(1/f - 1/g)
f,b,g,V
```

```{code-cell} ipython3
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path'
    },

    'rect' : {
        'type': 'rectangle',
        'emitter': {
            'type': 'telecentric_area_light',
            'radiance': mi.Color3f(1.0,1.0,1.0),
            'acc_angle': 25.0
    },
        'flip_normals' : True,
        'to_world': mi.Transform4f.translate(mi.Point3f(0,0,g+4))@mi.Transform4f.scale(mi.Point3f(2,2,2))
        
    },
    
    
          'sphere' : {
        'type': 'sphere',
              'bsdf' : {
        'type' : 'dielectric'
    },
    'center': [0, 0, g],
    'radius': 1.5,
        
    },
    
    
    'sensor': {
        'type': 'color_schlieren_sensor',
        'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, 0],
                                                 target=[0, 0, 1],
                                                 up=[0, 1, 0]),
        'foc_len' : f,
        'img_dist': b,
        'sen_size': 10.0,
        'lens_rad': 10.0,
        'film': {'type': 'hdrfilm',
      'width': 100,
      'height': 100,
      'rfilter': {'type': 'gaussian'},
      'pixel_format': 'rgb',
      'component_format': 'float32'},
     'sampler': {'type': 'independent', 'sample_count': 32},
    }
})
```

```{code-cell} ipython3
image2 = mi.render(scene)
```

```{code-cell} ipython3
imshow(10*image2)
```

## Light field displays

+++

The following images (also shown in the lecture) are acquired for a light field display prototype with a total range of possible angles of $10^\circ$ in both directions.

+++

<img src="figures/3/lfgCalibResults.svg" style="max-height:40vh">

+++

How can one explain the bottom right image?
