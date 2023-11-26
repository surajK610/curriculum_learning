'''
Exactly the same script run as en_ewt-ud.py, so reusing that code w/
just file changed to ontonotes.py so that naming convention holds
'''

import sys
import argparse
import yaml

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')
import importlib

ud_dep = "en_ewt-ud"
en_ewt_ud = importlib.import_module(ud_dep)
main_ud = getattr(en_ewt_ud, 'main')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=str, help="path to config file")
  args = parser.parse_args()
  
  config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
  main_ud(config)