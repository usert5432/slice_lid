"""
A definition of a decorator that precomputes batches in concurrent processes.

C.f. `lstm_ee.data.data_generator.base.multiprocessed_cache_base`.
"""

from lstm_ee.data.data_generator.base.multiprocessed_cache_base \
    import MultiprocessedCacheBase

from .idata_decorator import IDataDecorator

class MultiprocessedCache(MultiprocessedCacheBase, IDataDecorator):
    # pylint: disable=C0115

    def __init__(self, dgen, workers = None):
        IDataDecorator         .__init__(self, dgen)
        MultiprocessedCacheBase.__init__(self, dgen, workers)

