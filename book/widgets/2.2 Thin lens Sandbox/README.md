# Thin Lens Simulation

Interactive browser visualization of paraxial ray tracing through a thin lens.
It shows the object and image planes, aperture clipping, sensor position,
magnification, blur-circle diameter, field of view, and image-side depth of
focus.

## Run

Open `thin-lens-sandbox.html` in a modern browser. The simulation has no build
step. Plot navigation loads D3 7.9.0 from jsDelivr, so the page needs network
access unless that dependency is hosted locally.

## Files

- `thin-lens-sandbox.html` contains the simulation model, renderer, controls,
  and translations.
- `style.css` contains the page layout and component styling.
- `image.png` and `image_rotated.png` are runtime test-pattern assets.

## Development

Keep optical calculations independent of the DOM and rendering code. Visual
constants belong in `CONFIG`; fixed numerical tolerances belong in `MATH`.
