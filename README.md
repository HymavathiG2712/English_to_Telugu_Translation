# English_to_Telugu_Translation

https://github.com/bert-nmt/bert-nmt/tree/update-20-10?tab=readme-ov-file

	•	PyTorch version == 1.5.0
	•	Python version == 3.6
	•	huggingface/transformers version == 3.5.0

git clone https://github.com/bert-nmt/bert-nmt
cd bert-nmt
git checkout update-20-10
pip install --editable .


https://github.com/facebookresearch/fairseq

	•	PyTorch version >= 1.10.0
	•	Python version >= 3.8

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./


Check the files in examples/translations/
cd examples/translation/
cd ../..

TEXT=examples/translation/telugu.en-te

fairseq-preprocess --source-lang en --target-lang te     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test     --destdir data-bin/en-te --workers 20


CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/en-te \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric



fairseq-generate data-bin/en-te \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe  | tee -a checkpoints/test.log

