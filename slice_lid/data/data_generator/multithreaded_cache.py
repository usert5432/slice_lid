"""
A definition of a decorator that precomputes batches in concurrent threads.

C.f. `lstm_ee.data.data_generator.base.multithreaded_cache_base`.
"""

from lstm_ee.data.data_generator.base.multithreaded_cache_base \
    import MultithreadedCacheBase

from .idata_decorator import IDataDecorator

class MultithreadedCache(MultithreadedCacheBase, IDataDecorator):
    # pylint: disable=C0115

    def __init__(self, dgen, workers = None):
        IDataDecorator        .__init__(self, dgen)
        MultithreadedCacheBase.__init__(self, dgen, workers)

