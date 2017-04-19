
import disk_analyzer
import radial_slicer

# TODO:
# - read in image
# - convert image to grayscale?
# - crop image to smallest dimension (aka square it) - not sure if this step is needed or if we can just use smallest dimension in disk_classifier when determining params
# - send image to disk_classifier - get center and radius
# - crop image from center to radius + ~10%?
# - send image to radial_slicer - get
