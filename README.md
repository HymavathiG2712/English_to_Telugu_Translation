# English_to_Telugu_Translation


Using Latest version of Fairseq.

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


Results:
<img width="1440" alt="Screen Shot 2024-05-08 at 7 57 47 PM" src="https://github.com/HymavathiG2712/English_to_Telugu_Translation/assets/122757491/3b6b1513-6160-4a94-9768-cb63407ba165">



Using Fairseq 0.9.0

https://github.com/bert-nmt/bert-nmt/tree/update-20-10?tab=readme-ov-file

Update the prepare.script

git clone https://github.com/bert-nmt/bert-nmt
cd bert-nmt
git checkout update-20-10
pip install --editable .


	•	PyTorch version == 1.5.0
	•	Python version == 3.6
	•	huggingface/transformers version == 3.5.0


cd examples/translation
bash prepare-en_tel.sh

cd bpc_en_te
cp ../makedataforbert.sh .
bash makedataforbert.sh en

cd ../../..
TEXT=examples/translation/bpc_en_te
src=en
tgt=te
destdir=dest_en_te
python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $destdir  --joined-dictionary --bert-model-name bert-base-multilingual-cased


git clone https://github.com/pytorch/fairseq
git checkout a8f28ecb63ee01c33ea9f6986102136743d47ec2



CUDA_VISIBLE_DEVICES=0 python train.py \
    fairseq-0.9.0/dest_en_te \
    --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --no-epoch-checkpoints | tee -a fairseq-0.9.0/checkpoints/training.log


python generate.py  /N/u/ /Quartz/fairseq-0.9.0/dest_en_te \
--path /N/u/ /Quartz/fairseq-0.9.0/checkpoints/checkpoint_best.pt \
--batch-size 128 \
--beam 5 \
--remove-bpe | tee -a /N/u/ /Quartz/fairseq-0.9.0/checkpoints/nmt_test.log


NMT Model Training and evaluation done


src=en
tgt=te
bedropout=0.5
ARCH=transformer_iwslt_de_en
DATAPATH=/N/u/ /Quartz/fairseq-0.9.0/dest_en_te
SAVEDIR=checkpoints/bpcc${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp /N/u/ /Quartz/fairseq-0.9.0/checkpoints/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

export CUDA_VISIBLE_DEVICES=${1:-0}
python train.py $DATAPATH \
    -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
    --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
    --encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
    --bert-model-name bert-base-multilingual-cased | tee -a $SAVEDIR/training.log

Results:

<img width="1440" alt="Screen Shot 2024-05-08 at 7 56 34 PM" src="https://github.com/HymavathiG2712/English_to_Telugu_Translation/assets/122757491/7a9d1cae-3c45-4413-8818-cc2cc4fcd52a">

