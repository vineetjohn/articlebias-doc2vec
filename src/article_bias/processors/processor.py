import abc


class Processor(object):

    @abc.abstractmethod
    def process(self, **kwargs):
        return
