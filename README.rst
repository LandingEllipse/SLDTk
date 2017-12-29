================================
The Solar Limb Darkening Toolkit
================================

SLDTk is a commandline program for fast and robust modelling and correction
of limb darkening in solar images.


Getting Started
===============
Prerequisites
-------------
SLDTk is written for Python 3. The following packages are required:

- ``numpy>=1``
- ``opencv-python>=3``
- ``matplotlib>=2``

Note that in addition to ``opencv-python``, the underlying OpenCV library must
be installed. There are currently plans to include an optional dependency on
``scikit-image`` for systems without OpenCV.

Installation
------------
SLDTk can be installed using pip:

.. code-block:: console

    $ sudo pip install sldtk

Usage
-----

.. code-block:: console

    $ sldtk -i IMAGE [optional arguments]

The program takes a range of optional arguments for everything from detection
threshold to output directory. A complete list can be seen by calling

.. code-block:: console

    $ sldtk -h

Features
========
For now, the codebase provides the best source of information on SLDTk's
features and implementation specific details, as auto-generation of
documentation has not yet been set up.

Input Formats
-------------
Currently only images in the ``jpg`` and ``png`` formats are supported.

Detection
---------
The position and size of the solar disk is automatically determined using
computer vision. The disk is subsequently unwrapped and cleaned of outliers
(e.g. sunspots and facula) before an average intensity profile is generated.

Modelling
---------
A pluggable modelling system allows user specified models (e.g. a 2nd degree
polynomial) to be fitted to the intensity profile of the solar disk, and the
results to be plotted together with the original profile for visual analysis.
Reference models with known coefficients can also be overlaid on the produced
graph.

Correction
----------
Using a fitted model, the solar disk is flat-field corrected to a user
selectable bias point. The result can further be fed back into the pipeline
to assess its flatness, with a linear model fitted and plotted together
with the corrected intensity profile.

Contributing
============
Do you have an idea for how to make SLDTk better? Have you found a bug? Head
over to the `issue tracker <https://github.com/Legendin/SLDTk/issues>`_ to
open a new issue or contribute to an existing discussion.

If you are interested in contributing to the codebase,
`fork SLDTk on GitHub <https://github.com/Legendin/SLDTk#fork-destination-box>`_
to get started. You can install the project in development mode by using
``$ sudo pip install -e .`` from the cloned project root.

Have you found a solution to an existing issue or added a new feature?
`Pull requests <https://github.com/Legendin/SLDTk/pulls>`_ are always welcome.

License
=======
SLDTk is released under the MIT open source license. Please see
`LICENSE.txt <https://github.com/Legendin/SLDTk/blob/master/LICENSE.txt>`_
for details.

Acknowledgments
===============
The Solar Limb Darkening Toolkit began as an experimental physics project
together with Matt Wingham and Sam Heron at Aberystwyth University in the
fall of 2017. It is written and maintained by Ariel Ladegaard.
