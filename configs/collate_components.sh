for i in {0..11}; do python3 src/collate_metrics.py --exp fpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp cpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp dep --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp fpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp cpos --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp dep --dataset en_ewt-ud --metric "Val Acc" --resid False --plot none

for i in {0..11}; do python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc" --resid False --plot none

for i in {0..11}; do python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "Val Acc" --resid False --plot none --attention-head $i; done
for i in {0..11}; do python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "Val Acc" --resid False --plot none --attention-head $i; done

python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "Val Acc" --resid False --plot none
python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "Val Acc" --resid False --plot none


