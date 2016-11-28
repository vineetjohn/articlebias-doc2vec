import getopt
import sys

from src.article_bias.processors.model_trainer import ModelTrainer
from src.article_bias.processors.amazon_processor import AmazonProcessor
from src.article_bias.processors.article_classifier import ArticleClassifier


def main(argv):

    mode = None
    labeled_articles_source_file_path = None
    doc2vec_model_file_path = None
    ml_model_file_path = None
    articles_source_file_path = None

    usage_msg = \
        "usage : article_bias.py " + \
        "--mode [training|evaluation] " + \
        "[--labeled_articles_source_file <labeled_articles_source_file_path>] " + \
        "--doc2vec_model_file_path <doc2vec_model_file_path> " + \
        "--ml_model_file_path <ml_model_file_path> " + \
        "[--articles_source_file_path <articles_source_file_path>] "

    try:
        opts, args = \
            getopt.getopt(
                args=argv,
                shortopts='h',
                longopts=['mode=', 'labeled_articles_source_file_path=', 'doc2vec_model_file_path=',
                          'ml_model_file_path=', 'articles_source_file_path=']
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
        elif opt == "--output_file_path":
            output_file_path = arg

    validate_arguments_and_process(
        usage_msg, mode, labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
        articles_source_file_path
    )


def validate_arguments_and_process(usage_msg, mode, labeled_articles_source_file_path, doc2vec_model_file_path,
                                   ml_model_file_path, articles_source_file_path):

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

    model_trainer = \
        ArticleClassifier(
            labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
            articles_source_file_path
        )
    model_trainer.process()

if __name__ == "__main__":
    main(sys.argv[1:])
