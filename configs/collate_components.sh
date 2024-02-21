# ----------------------------- EN_EWT-UD ---------------------------- #
for i in {0..11}; do python3 src/collate_metrics.py --exp fpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp cpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp dep --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp fpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp cpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp dep --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none


python3 scripts/make_multiheatmaps.py --folder_path outputs/en_ewt-ud/cpos/components --output_path figures/aheads/cpos_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/en_ewt-ud/fpos/components --output_path figures/aheads/fpos_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/en_ewt-ud/dep/components --output_path figures/aheads/dep_components.png

# ------------------------------- ONTONOTES ------------------------------- #

for i in {0..11}; do python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc" --resid False --plot none

python3 scripts/make_multiheatmaps.py --folder_path outputs/ontonots/ner/components --output_path figures/ontonotes/ner_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/ontonots/phrase_start/components --output_path figures/ontonotes/phrase_start_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/ontonots/phrase_end/components --output_path figures/ontonotes/phrase_end_components.png


# ------------------------------- PTB_3 ------------------------------- #

for i in {0..11}; do python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "Root Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "NSpr" --resid False --plot none --attention-head $i; done

for i in {0..11}; do python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "UUAS" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "DSpr" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "Root Acc" --resid False --plot none
python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "NSpr" --resid False --plot none

python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "UUAS" --resid False --plot none
python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "DSpr" --resid False --plot none

python3 scripts/make_multiheatmaps.py --folder_path outputs/ptb_3/depth/components/NSpr --output_path figures/ptb_3/depth_nspr_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/ptb_3/depth/components/Root_Acc --output_path figures/ptb_3/depth_root_acc_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/ptb_3/distance/components/DSpr --output_path figures/ptb_3/distance_dspr_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/ptb_3/distance/components/UUAS --output_path figures/ptb_3/distance_uuas_components.png


# ------------------------------- AHEADS ------------------------------- #

for i in {0..11}; do python3 src/collate_metrics.py --exp previous_token_head --dataset aheads --metric "Val Acc" --resid False --plot none --attention-head $i; done
python3 src/collate_metrics.py --exp previous_token_head --dataset aheads --metric "Val Acc" --resid False --plot none

for i in {0..11}; do python3 src/collate_metrics.py --exp duplicate_token_head --dataset aheads --metric "Val Acc" --resid False --plot none --attention-head $i; done
python3 src/collate_metrics.py --exp duplicate_token_head --dataset aheads --metric "Val Acc" --resid False --plot none

for i in {0..11}; do python3 src/collate_metrics.py --exp induction_head --dataset aheads --metric "Val Acc" --resid False --plot none --attention-head $i; done
python3 src/collate_metrics.py --exp induction_head --dataset aheads --metric "Val Acc" --resid False --plot none

python3 scripts/make_multiheatmaps.py --folder_path outputs/aheads/induction_head/components --output_path figures/aheads/induction_head_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/aheads/previous_token_head/components --output_path figures/aheads/previous_token_head_components.png
python3 scripts/make_multiheatmaps.py --folder_path outputs/aheads/duplicate_token_head/components --output_path figures/aheads/duplicate_token_head_components.png
