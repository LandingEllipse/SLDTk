# ssr-ph255-solar-studies python limb darkening toolkit

## Components in order of invocation:
- Disk Analyzer
  * finds sun centre
  * finds sun radius

- Sun Slicer
  * takse N slices from the centre of the solar disk to its radius, returns a stack of slices as np-arrays

- Sanitizer
  * drops slices containing sunspots (and potentially also "whitespots"?)

- Fitter
  * does non-linear regression to find coefficients (a_k)

- Plotter
  * plots mean, median and std of the slice stack, as well as a fitted curve and coefficients overlay
 
- Flattener
  * Uses the found coefficients and some yet-to-be-determined algorithm to flatten the input image.
  
  
## Challenges
Single channel HMI intensitygrams downloadable from SDO have been converted in a non-standard way, as evident by the fact that if we convert a multi channel intensitygram from the same source with e.g. `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`, the result (and hence the intensity curve) is distinctly different. It appears, then, that the former have not been converted in accordance with the ITU-R 601-2 luma transform (i.e. `L = R * 299/1000 + G * 587/1000 + B * 114/1000`), but the exact method used remains unknown.
The consequence is that We will likely have to think twice about how we verify limb darkening coefficients, and how we compare between SDO samples and e.g. local ground based samples.
