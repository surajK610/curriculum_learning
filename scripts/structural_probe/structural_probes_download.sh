# wget https://nlp.stanford.edu/~johnhew/public/en_ewt-ud-sample.tgz

# tar xzvf en_ewt-ud-sample.tgz
# mkdir -p data/structural-probes
# mv en_ewt-ud-sample data/structural-probes
# rm en_ewt-ud-sample.tgz

wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu -P data/structural-probes
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -P data/structural-probes
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu -P data/structural-probes

