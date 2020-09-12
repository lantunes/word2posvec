from word2posvec import TrainingData, Trainer


if __name__ == '__main__':
    td = TrainingData.load("brown.top5k.training.data")

    words, tags = td.to_one_hot()

    tr = Trainer(dim_in=2, dim_out=len(tags[0]))

    embeddings = tr.train(words, tags, step_size=0.01, num_epochs=10, batch_size=256)

    Trainer.save_embeddings(embeddings, "brown.top5k.dim2.embeddings")
