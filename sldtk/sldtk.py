#!/usr/bin/env python3
"""

TODO
----
Change all single quotes to double quotes.

"""
import os

import cv2

from . import correction
from . import detection
from . import models
from . import plotting
from . import profile
from .helpers import (
    parse_input,
    generate_output_paths,
    overlay_mec,
    plot_correction
)

config = {
    "debug": False,
    "operations": ['all', 'correct', 'model'],
    "threshold": 10,
    "slices": 1000,
    "bias": 175,
    "models": list(models.models.keys()),
    "reference_models": list(models.reference_models.keys()),
    "plot_correction": True,
    "interactive_plot": False,
    "out_dir": "./out",
    "debug_dir": None,
    "separate_dir": True,
}


def main():
    args = parse_input(config)
    paths = generate_output_paths(args)

    image = cv2.imread(args['image'])
    if image is None:
        raise TypeError(
            "{} not recognized as a jpg or png image.".format(args['image']))

    if image.ndim > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect the solar disk.
    disk_attr = detection.detect_disk(gray, args['threshold'])
    if args['debug']:
        print("MEC x: {}, y: {}, r: {}".format(disk_attr[0], disk_attr[1],
                                               disk_attr[2]))
        image = overlay_mec(image, disk_attr)
        cv2.imwrite(paths["mec"], image, (cv2.IMWRITE_JPEG_QUALITY, 50,
                                          cv2.IMWRITE_PNG_COMPRESSION, 6))

    # Create the slice stack.
    stack = profile.extract_stack(gray, disk_attr, args['slices'])
    stack_clean = profile.clean_stack(stack)
    # Average the stack to create an intensity profile.
    intensity_profile = profile.compress_stack(stack_clean)
    if args['debug']:
        print("Slices: {}".format(len(stack)))
        print("Slices dropped: {}".format(len(stack) - len(stack_clean)))
        stack = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
        stack = cv2.line(stack, (disk_attr[2]-1, 0),
                         (disk_attr[2]-1, stack.shape[0]), (0, 255, 0))
        cv2.imwrite(paths['stack'], stack)
        print("Slice stack saved to {}".format(paths['stack']))
        stack_clean = cv2.cvtColor(stack_clean, cv2.COLOR_GRAY2BGR)
        stack_clean = cv2.line(stack_clean, (disk_attr[2]-1, 0),
                               (disk_attr[2]-1, stack_clean.shape[0]),
                               (0, 255, 0))
        cv2.imwrite(paths['stack_clean'], stack_clean)
        print("Clean slice stack saved to {}".format(paths['stack_clean']))

    model = models.models[args["model"]]()
    model.fit(intensity_profile, args["model_parameter"])
    print("Model coefficients: {}".format(model.coefs_str()))

    corrected = None
    if args['operation'] in ('all', 'correct'):
        # Apply flat-field correction.
        corrected = correction.correct_disk(gray, disk_attr, args['bias'],
                                            model)
        cv2.imwrite(paths['corrected'], corrected)
        print("Corrected image saved to {}".format(paths['corrected']))

    if args['operation'] in ('all', 'model'):
        # Plot intensity profile together with computed model.
        img_name = os.path.basename(args['image'])
        plotter = plotting.Plotter(img_name, paths['plot'])
        plotter.plot_profile(intensity_profile, zorder=2)
        plotter.plot_model("Fitted", model, zorder=3)

        if args["reference_model"] is not None:
            ref = models.reference_models[args["reference_model"]]
            reference_model = ref[0]()
            reference_model.coefs = ref[2]
            plotter.plot_model(ref[1], reference_model, zorder=2, color='g',
                               linestyle=':')

        if args['plot_correction'] and corrected is not None:
            plot_correction(corrected, disk_attr, args, plotter)

        if args['interactive_plot']:
            plotter.show()
        else:
            plotter.save()
            print("Intensity profile plot saved to {}".format(paths['plot']))


if __name__ == "__main__":
    main()


