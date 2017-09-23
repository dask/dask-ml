import dask.array as da
from distributed.diagnostics.plugin import SchedulerPlugin

def memoize():
    """
    We want to support memoization somehow...

    """


class CachingPlugin(SchedulerPlugin):

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.add_plugin(self)

    def transition(self, key, start, finish, nbytes=None, startstops=None,
                   *args, **kwrags):
        if start == 'processing' and finish == 'memory':
            self.scheduler.client_desires_keys(keys=[key],
                                               client='fake-caching-client')
        no_longer_desired_keys = self.cleanup()
        self.scheduler.client_releases_keys(keys=no_longer_desired_keys,
                                            client='fake-caching-client')

    def cleanup(self):
        return {}
