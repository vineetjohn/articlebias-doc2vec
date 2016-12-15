import getopt
import sys

from processors.amazon_line_processor_tfidf import AmazonLineProcessorTfIdf
from processors.amazon_line_processor_bigram import AmazonLineProcessorBigram
from processors.fact_check_processor_doc2vec import FactCheckProcessor
from processors.fact_check_processor_tfidf import FactCheckProcessorTFIDF


def main(argv):

    mode = None
    labeled_articles_source_file_path = None
    doc2vec_model_file_path = None
    ml_model_file_path = None
    articles_source_file_path = None
    shuffle_count = None
    classification_sources_file_path = None

    usage_msg = \
        "usage : article_bias.py " + \
        "--mode [training|evaluation] " + \
        "[--labeled_articles_source_file <labeled_articles_source_file_path>] " + \
        "--doc2vec_model_file_path <doc2vec_model_file_path> " + \
        "--ml_model_file_path <ml_model_file_path> " + \
        "--shuffle_count <shuffle_count> " + \
        "[--articles_source_file_path <articles_source_file_path>] " + \
        "--classification_sources_file_path"

    try:
        opts, args = \
            getopt.getopt(
                args=argv,
                shortopts='h',
                longopts=['mode=', 'labeled_articles_source_file_path=', 'doc2vec_model_file_path=',
                          'ml_model_file_path=', 'shuffle_count=', 'articles_source_file_path=',
                          'classification_sources_file_path=']
            )
    except getopt.GetoptError as e:
        print(e)
        print(usage_msg)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            sys.exit()
        elif opt == "--mode":
            mode = arg
        elif opt == "--labeled_articles_source_file_path":
            labeled_articles_source_file_path = arg
        elif opt == "--doc2vec_model_file_path":
            doc2vec_model_file_path = arg
        elif opt == "--ml_model_file_path":
            ml_model_file_path = arg
        elif opt == "--articles_source_file_path":
            articles_source_file_path = arg
        elif opt == "--shuffle_count":
            shuffle_count = arg
        elif opt == "--classification_sources_file_path":
            classification_sources_file_path = arg

    validate_arguments_and_process(
        usage_msg, mode, labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
        articles_source_file_path, shuffle_count, classification_sources_file_path
    )


def validate_arguments_and_process(usage_msg, mode, labeled_articles_source_file_path, doc2vec_model_file_path,
                                   ml_model_file_path, articles_source_file_path, shuffle_count,
                                   classification_sources_file_path):

    if not mode:
        print(usage_msg)
        raise RuntimeError("No mode specified")

    if not labeled_articles_source_file_path:
        print(usage_msg)
        raise RuntimeError("No labeled_articles_source_file specified")

    if not doc2vec_model_file_path:
        print(usage_msg)
        raise RuntimeError("No doc2vec_model_file_path specified")

    if not ml_model_file_path:
        print(usage_msg)
        raise RuntimeError("No ml_model_file_path specified")

    if not articles_source_file_path:
        print(usage_msg)
        raise RuntimeError("No articles_source_file_path specified")

    if not shuffle_count:
        print(usage_msg)
        raise RuntimeError("No shuffle_count specified")

    model_trainer = \
        FactCheckProcessorTFIDF(
            labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
            articles_source_file_path, int(shuffle_count), classification_sources_file_path
        )
    model_trainer.process()

if __name__ == "__main__":
    main(sys.argv[1:])
