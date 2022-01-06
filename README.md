# tiny-hintguess
Tiny hint guess game 

First install environment using (requires `torch 1.10` and above):
`pip install -r requirements.txt`


To train agents, use
`python train_agents.py --config-file configs/<your_own_config>.yaml`

Or use (if you want to train agents in parallel batches):
`sbatch ./batch.sh`
