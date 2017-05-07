import os
import argparse
import cv2

import disk_detection
import profile_generation
import correction
import plotting
from models import Polynomial

config = {
    "operations": ['all', 'correct', 'model'],
    "threshold": 10,
    "slices": 100,
    "debug": False,
    "cmode": ['profile', 'model'],
    "bias": 128,
}


def generate_output_paths(args):
    basename = os.path.basename(args['image'])
    root, ext = os.path.splitext(basename)
    out_dir = "./out"
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        'intensity': "{}/{}_intensity{}".format(out_dir, root, ext),
        'intensity_txt': "{}/{}_intensity.csv".format(out_dir, root),
        'corrected': "{}/{}_corrected_{}_{}{}".format(out_dir, root, args['bias'], args['cmode'], ext),
        'plot': "{}/{}_plot.png".format(out_dir, root)
    }

    if args['debug']:
        print("Debug mode active.")
        debug_dir = "./out/debug"
        os.makedirs(debug_dir, exist_ok=True)
        paths['debug_stack'] = "{}/{}_stack{}".format(debug_dir, root, ext)
        paths['debug_stack_clean'] = "{}/{}_stack_clean{}".format(debug_dir, root, ext)
        paths['debug_mec'] = "{}/{}_mec{}".format(debug_dir, root, ext)

    return paths


def pos_mul_4_int(arg):
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("{} could not be interpreted as an integer.".format(arg))
    else:
        if val < 0 or val % 4 != 0:
            raise argparse.ArgumentTypeError("{} is not a positive, even integer.".format(val))
    return val


def uint8(arg):
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("{} could not be interpreted as an integer.".format(arg))
    else:
        if val < 0 or val > 255:
            raise argparse.ArgumentTypeError("{} can not be represented as an unsigned 8-bit integer.".format(val))
    return val


def parse_input():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to a jpg or png solar image file")
    ap.add_argument("-o", "--operation", choices=config["operations"], default=config["operations"][0], help="the operation that should be performed on the image")
    ap.add_argument("-c", "--cmode", choices=config["cmode"], default=config["cmode"][0], help="the correction mode to use")
    ap.add_argument("-s", "--slices", type=pos_mul_4_int, default=config["slices"], help="number of slices to average to create the intensity profile")
    ap.add_argument("-t", "--threshold", type=uint8, default=config["threshold"], help="brightness threshold for the solar disk (uint8)")
    ap.add_argument("-b", "--bias", type=uint8, default=config["bias"], help="brightness bias for the correction (uint8)")
    ap.add_argument("-d", "--debug", type=bool, default=config['debug'], help="if enabled, provides intermediary output to the 'debug' directory")
    args = vars(ap.parse_args())

    if not os.path.isfile(args['image']):
        raise argparse.ArgumentTypeError("{} is not a path to a file.".format(args['image']))

    image = cv2.imread(args['image'])
    if image is None:
        raise TypeError("{} is not a valid jpg or png image.".format(args['image']))

    return args, image


def main():
    args, image = parse_input()
    out_paths = generate_output_paths(args)

    if len(image.shape) > 2:  # color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect the solar disk
    disk_attr = disk_detection.detect_disk(gray, args['threshold'])
    if args['debug']:
        print("Min enclosing circle x: {}, y: {}, r: {}".format(disk_attr[0], disk_attr[1], disk_attr[2]))
        cv2.circle(image, (disk_attr[0], disk_attr[1]), disk_attr[2], (0, 255, 0), 6)
        cv2.rectangle(img=image, pt1=(disk_attr[0] - 10, disk_attr[1] - 10), pt2=(disk_attr[0] + 10, disk_attr[1] + 10), color=(0, 255, 0), thickness=-1)
        cv2.imwrite(out_paths['debug_mec'], image)

    # Create a slice stack
    stack = profile_generation.create_slice_stack(gray, disk_attr, args['slices'])
    stack_clean = profile_generation.reject_outliers(stack)  # TODO: add 'gradient' arg for sensitivity?
    # Average the stack to create an intensity profile
    intensity_profile = profile_generation.compress_stack(stack_clean)
    if args['debug']:
        # Slicing
        print("Slices: {}".format(len(stack)))
        print("Slices dropped: {}".format(len(stack) - len(stack_clean)))
        cv2.imwrite(out_paths['debug_stack'], stack)
        print("Slice stack saved to {}".format(out_paths['debug_stack']))
        cv2.imwrite(out_paths['debug_stack_clean'], stack_clean)
        print("Clean slice stack saved to {}".format(out_paths['debug_stack_clean']))

    # Attempt to model the data as a polynomial
    if args['operation'] in ('all', 'model') or args['cmode'] == 'model':
        model = Polynomial()
        model.fit(intensity_profile, {'degree': 2})
        print("Model fitted with coefficients: a0: {}, a1: {}, a2: {}".format(*model.coefs))

    # Plot intensity profile together with computed model
    if args['operation'] in ('all', 'model'):
        img_name, _ = os.path.splitext(os.path.basename(args['image']))
        plotter = plotting.Plotter(img_name, out_paths['plot'])
        plotter.plot_intensity_profile(intensity_profile)
        plotter.plot_model("Fitted", model, zorder=3)

        reference = Polynomial()
        reference.coefs = (0.3, 0.93, -0.23)  # TEMP: 550nm reference curve
        plotter.plot_model("550nm", reference, zorder=2, color='g', linestyle=':')

        # plotter.show()
        plotter.save()
        print("Intensity profile plot saved to {}".format(out_paths['plot']))

    # Apply flat-field correction
    if args['operation'] in ('all', 'correct'):
        if args['cmode'] == 'profile':
            corrected = correction.correct_disk(gray, disk_attr, args['bias'], profile=intensity_profile)
        else:  # 'model'
            corrected = correction.correct_disk(gray, disk_attr, args['bias'], model=model)
        cv2.imwrite(out_paths['corrected'], corrected)
        print("Corrected image saved to {}".format(out_paths['corrected']))


if __name__ == "__main__":
    main()

# path1 = "tests/images/20170315_130000_4096_HMIIC_-watermark_small.jpg"
# path2 = "tests/images/20170315_130000_4096_HMIIC_-watermark_small_offset.jpg"
# path3 = "tests/images/20170315_aberystwyth_combined.jpg"
# path4 = "tests/images/20170315_aberystwyth_combined_square.jpg"
# path5 = "tests/images/LimbDark.png"
# path6 = "tests/images/LimbDark_marked_square.png"
# path7 = "tests/images/20170420_4096_HMIIC.jpg"
# path8 = "tests/images/20170420_4096_HMII.jpg"
# path9 = "tests/images/20170315_125238_4096_HMII_small.jpg"
# path10 = "tests/images/20140704_022325_4096_HMII.jpg"
