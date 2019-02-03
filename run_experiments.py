from slang.corpus_processing import CorpusProcessing
from slang.trainer import Trainer


def main():
    # CoNLL2003
    connl2003 = CorpusProcessing(corpus='connl2003')
    connl2003.process("datasets/CoNLL2003/")
    # connl2003.load_embeddings('glove.6B.100d.txt', emb_type='glove')
    trainer = Trainer(connl2003)
    trainer.train_and_evaluate(epochs=2)

    # ToDo: generate data on the fly, memory consumption
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    # Paramopama
    # paramopama = CorpusProcessing(corpus='paramopama')
    # paramopama.process("datasets/Paramopama/")
    # # paramopama.convert_tags()
    # paramopama.split_corpus(split=0.6)
    # paramopama.load_embeddings('embeddings/CHAVE/CHAVE_word2vec.keyed', emb_type='word2vec')
    # #paramopama.load_embeddings('embeddings/publico_vectors_non-breaking-spaces.bin', emb_type='word2vec')
    # trainer = Trainer(paramopama)
    # trainer.train_and_evaluate(epochs=1)

    # CINTIL
    # cintil = CorpusProcessing(corpus='cintil')
    # cintil.process("datasets/CINTIL/")
    # cintil.split_corpus(split=0.8)
    # trainer = Trainer(cintil)
    # trainer.train_and_evaluate(epochs=1)

    # Comtravo
    # comtravo = CorpusProcessing(corpus='comtravo')
    # comtravo.process("datasets/Comtravo/")
    # print(len(comtravo.train_tokens))
    # print(len(comtravo.train_tags))
    # print(comtravo.max_sent_length)
    # print(comtravo.max_word_length)

    # trainer = Trainer(comtravo)
    # trainer.cross_fold_evaluation(epochs=5)


if __name__ == "__main__":
    main()
