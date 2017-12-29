import argparse
import os

import cv2

from . import models
from . import profile


def plot_correction(img, disk_attr, args, plotter):
    stack = profile.extract_stack(img, disk_attr, args['slices'])
    stack_clean = profile.clean_stack(stack)
    intensity_profile = profile.compress_stack(stack_clean)
    model = models.Linear()
    model.fit(intensity_profile)
    print("Linearity of correction: {}".format(model.coefs_str()))
    # TODO: implement kwargs forwarding for plot_profile?
    plotter.plot_profile(intensity_profile, zorder=1, color='brown',
                         label="Corrected profile")
    plotter.plot_model("Corrected", model, zorder=1, color='cyan')


def overlay_mec(img, disk_attr, color=(0, 255, 0)):
    x, y, r = disk_attr
    thickness = int(round(r/200))  # Reasonable thickness for different sizes.

    cv2.circle(img, (x, y), r, color, thickness)
    cv2.rectangle(img=img, pt1=(x-thickness, y-thickness),
                  pt2=(x+thickness, y+thickness), color=color, thickness=-1)

    cv2.putText(img,
                text='x: {}, y: {}, r: {}'.format(*disk_attr),
                org=(20, img.shape[0]-20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=thickness/2,
                color=color,
                thickness=thickness)
    return img


def _pos_int(arg):
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("{} could not be interpreted as an "
                                         "integer.".format(arg))
    else:
        if val <= 0:
            raise argparse.ArgumentError(val, "must be a positive integer.")
    return val


def _uint8(arg):
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("{} could not be interpreted as an "
                                         "integer.".format(arg))
    else:
        if val < 0 or val > 255:
            raise argparse.ArgumentError(val,
                                         "must be within the range (0, 255)")
    return val


def _str2bool(arg):
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_input(config):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Model and correct for limb darkening in a solar image.")
    ap.add_argument("-i", "--image",
                    required=True,
                    help="Path to a jpg or png solar image file.")
    ap.add_argument("-o", "--operation",
                    choices=config["operations"],
                    default=config["operations"][0],
                    help="Operation to perform.")
    ap.add_argument("-s", "--slices",
                    type=_pos_int,
                    default=config["slices"],
                    help="Number of slices to average to create the intensity "
                         "profile.")
    ap.add_argument("-t", "--threshold",
                    type=_uint8,
                    default=config["threshold"],
                    help="Brightness threshold for the solar disk (uint8).")
    ap.add_argument("-b", "--bias",
                    type=_uint8,
                    default=config["bias"],
                    help="Brightness bias for the correction (uint8).")
    ap.add_argument("-d", "--debug",
                    type=_str2bool,
                    nargs="?",
                    const=True,
                    default=config["debug"],
                    help="Provide intermediary output to the debug directory.")
    ap.add_argument("-m", "--model",
                    choices=config["models"],
                    default=config["models"][0],
                    help="How to model the limb darkening.")
    ap.add_argument("-p", "--model_parameter",
                    help="Model parameters, e.g. degree for polynomial.")
    ap.add_argument("-r", "--reference_model",
                    choices=config["reference_models"],
                    const=None,
                    help="Whether and which reference model to plot.")
    ap.add_argument("-P", "--plot_correction",
                    type=_str2bool,
                    nargs="?",
                    const=True,
                    default=config["plot_correction"],
                    help="Feed correction back to assess flatness.")
    ap.add_argument("-I", "--interactive_plot",
                    type=_str2bool,
                    nargs="?",
                    const=True,
                    default=config["interactive_plot"],
                    help="Show instead of saving plots.")
    ap.add_argument("-S", "--separate_dir",
                    type=_str2bool,
                    nargs="?",
                    const=True,
                    default=config["separate_dir"],
                    help="Generate separate output directories per image.")
    ap.add_argument("--out_dir",
                    default=config["out_dir"],
                    help="Path to a custom output directory.")
    ap.add_argument("--debug_dir",
                    default=config["debug_dir"],
                    help="Path to a custom debug directory.")
    args = vars(ap.parse_args())

    if not os.path.isfile(args['image']):
        raise argparse.ArgumentTypeError(
            "{} is not a path to a file.".format(args['image']))

    return args


def generate_output_paths(args):
    basename = os.path.basename(args['image'])
    root, ext = os.path.splitext(basename)
    out_dir = args["out_dir"]
    if args["separate_dir"]:
        out_dir = os.path.join(out_dir, root)
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        'intensity': "{}/{}_intensity{}".format(out_dir, root, ext),
        'corrected': "{}/{}_corrected_{}{}".format(out_dir, root, args['bias'],
                                                   ext),
        'plot': "{}/{}_plot.png".format(out_dir, root)
    }

    if args['debug']:
        print("Debug mode active.")
        debug_dir = args["debug_dir"] or args["out_dir"]
        if args["separate_dir"]:
            debug_dir = os.path.join(debug_dir, root)
        os.makedirs(debug_dir, exist_ok=True)
        paths['stack'] = "{}/{}_stack{}".format(debug_dir, root, ext)
        paths['stack_clean'] = "{}/{}_stack_clean{}".format(debug_dir, root,
                                                            ext)
        paths['mec'] = "{}/{}_mec{}".format(debug_dir, root, ext)

    return paths