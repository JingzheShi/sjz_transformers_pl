I created an implementation of the basic transformer model using pytorch-lightning according to the article *attention is all you need*.

To train, run 

```bash
python3 train.py
```

To use, create an input.txt in the same dir and run

```bash
python3 use.py
```

Some arguments of `train.py` can be chosen, which is written in `train.py` and `model/my_tf.py`.