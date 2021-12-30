The implementation of [Parameter Differentiation based Multilingual Neural Machine Translation](https://arxiv.org/abs/2112.13619).


# Requirements

```
pip install fairseq==0.10.2
conda install scikit-learn
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

# Usage

1. Prepare data following [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/translation#multilingual-translation):

```
unzip data-bpe.zip

mkdir -p data-bin && cut -f1 data-bpe/bpe.vocab | tail -n +4 | sed "s/$/ 100/g" > data-bin/dict.en.txt

for lang in es pt; do
    fairseq-preprocess --source-lang en --target-lang $lang \
        --trainpref data-bpe/train.en-$lang \
        --validpref data-bpe/valid.en-$lang \
        --testpref  data-bpe/test.en-$lang  \
        --destdir data-bin \
        --srcdict data-bin/dict.en.txt \
        --tgtdict data-bin/dict.en.txt 
done

```


2. Training:


Multilingual NMT:
```
fairseq-train data-bin --user-dir . --max-tokens 4096 --max-update 20000 \
    --task multilingual_translation --lang-pairs es-en,pt-en  \
    --arch parameter_differentiation_tiny_model --share-all-embeddings --share-encoders --share-decoders  \
    --lr-scheduler inverse_sqrt --optimizer adam --lr 0.0015 --validate-interval 4
```


Parameter differentiation based MNMT
```
fairseq-train data-bin --user-dir . --max-tokens 4096 --max-update 20000  \
    --task parameter_differentiation_task --lang-pairs es-en,pt-en  \
    --arch parameter_differentiation_tiny_model --share-all-embeddings  \
    --lr-scheduler inverse_sqrt --optimizer adam --lr 0.0015 --validate-interval 4
```


3. Decoding
```
fairseq-generate data-bin --user-dir . --max-tokens 4096 --quiet \
    --task parameter_differentiation_task --lang-pairs es-en,pt-en \
    --remove-bpe sentencepiece --source-lang es --target-lang en \
    --path checkpoints/checkpoint_last.pt

fairseq-generate data-bin --user-dir . --max-tokens 4096 --quiet \
    --task parameter_differentiation_task --lang-pairs es-en,pt-en \
    --remove-bpe sentencepiece --source-lang pt --target-lang en \
    --path checkpoints/checkpoint_last.pt
```



