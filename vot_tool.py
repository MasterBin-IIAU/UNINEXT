"""
\file vot.py

@brief Python utility functions for VOT toolkit integration

@author Luka Cehovin, Alessio Dore

@date 2023

"""

import os
import collections
import numpy as np

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])
Empty = collections.namedtuple('Empty', [])

class VOT(object):
    """ Base class for VOT toolkit integration in Python.
        This class is only a wrapper around the TraX protocol and can be used for single or multi-object tracking.
        The wrapper assumes that the experiment will provide new objects onlf at the first frame and will fail otherwise."""
    def __init__(self, region_format, channels=None, multiobject: bool = None):
        """ Constructor for the VOT wrapper.

        Args:
            region_format: Region format options
            channels: Channels that are supported by the tracker
            multiobject: Whether to use multi-object tracking
        """
        assert(region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON, trax.Region.MASK])

        if multiobject is None:
            multiobject = os.environ.get('VOT_MULTI_OBJECT', '0') == '1'

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels, metadata=dict(vot="python"), multiobject=multiobject)

        request = self._trax.wait()
        assert(request.type == 'initialize')

        self._objects = []

        assert len(request.objects) > 0 and (multiobject or len(request.objects) == 1)

        for object, _ in request.objects:
            if isinstance(object, trax.Polygon):
                self._objects.append(Polygon([Point(x[0], x[1]) for x in object]))
            elif isinstance(object, trax.Mask):
                self._objects.append(object.array(True))
            else:
                self._objects.append(Rectangle(*object.bounds()))

        self._image = [x.path() for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]

        self._multiobject = multiobject

        self._trax.status(request.objects)

    def region(self):
        """
        Returns initialization region for the first frame in single object tracking mode.

        Returns:
            initialization region
        """

        assert not self._multiobject

        return self._objects[0]

    def objects(self):
        """
        Returns initialization regions for the first frame in multi object tracking mode.

        Returns:
            initialization regions for all objects
        """

        return self._objects

    def report(self, status, confidence = None):
        """
        Report the tracking results to the client

        Arguments:
            status: region for the frame or a list of regions in case of multi object tracking
            confidence: confidence for the object detection, used only in single object tracking mode
        """

        def convert(region):
            """ Convert region to TraX format """
            # If region is None, return empty region
            if region is None: return trax.Rectangle.create(0, 0, 0, 0)
            assert isinstance(region, (Empty, Rectangle, Polygon, np.ndarray))
            if isinstance(region, Empty):
                return trax.Rectangle.create(0, 0, 0, 0)
            elif isinstance(region, Polygon):
                return trax.Polygon.create([(x.x, x.y) for x in region.points])
            elif isinstance(region, np.ndarray):
                return trax.Mask.create(region)
            else:
                return trax.Rectangle.create(region.x, region.y, region.width, region.height)

        if not self._multiobject:
            status = convert(status)
        else:
            assert isinstance(status, (list, tuple))
            status = [(convert(x), {}) for x in status]

        properties = {}

        if not confidence is None and not self._multiobject:
            properties['confidence'] = confidence

        self._trax.status(status, properties)

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        # Only the first frame can declare new objects for now
        assert request.objects is None or len(request.objects) == 0

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None

    def quit(self):
        """ Quit the tracker"""
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        """ Destructor for the tracker, calls quit. """
        self.quit()

class VOTManager(object):
    """ VOT Manager is provides a simple interface for running multiple single object trackers in parallel. Trackers should implement a factory interface. """

    def __init__(self, factory, region_format, channels=None):
        """ Constructor for the manager. 
        The factory should be a callable that accepts two arguments: image and region and returns a callable that accepts a single argument (image) and returns a region.

        Args:
            factory: Factory function for creating trackers
            region_format: Region format options
            channels: Channels that are supported by the tracker
        """
        self._handle = VOT(region_format, channels, multiobject=True)
        self._factory = factory

    def run(self):
        """ Run the tracker, the tracking loop is implemented in this function, so it will block until the client terminates the connection."""
        objects = self._handle.objects()

        # Process the first frame
        image = self._handle.frame()
        if not image:
            return

        trackers = [self._factory(image, object) for object in objects]

        while True:

            image = self._handle.frame()
            if not image:
                break

            status = [tracker(image) for tracker in trackers]

            self._handle.report(status)
