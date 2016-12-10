from gensim import utils
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument


class ClassifiedTaggedLineDocument(TaggedLineDocument):

    def __init__(self, source, label):
        super(ClassifiedTaggedLineDocument, self).__init__(source)
        self.label = label

    def __iter__(self):
        try:
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [self.label + str(item_no)])
        except AttributeError:
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [self.label + str(item_no)])
