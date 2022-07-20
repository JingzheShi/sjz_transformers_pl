I created an implementation of the basic transformer model using pytorch-lightning according to the article *attention is all you need*.

To train, run 

```bash
python3 train.py
```

To use, run

```bash
python3 use.py --relative_translation_source_path=$input_file
```

In which default `input_file` is `input.txt` in the same dir.



Some arguments of `train.py` can be adjusted, which is written in `train.py` and `model/my_tf.py`.