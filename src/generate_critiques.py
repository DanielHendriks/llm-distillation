import argparse
import logging
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.generation_utils import ProcessHandler 
from dataloaders.cqa import CQADatasetLoader

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'llm_outputs', 'rationales_critiqued')


class CritiqueProcessHandler(ProcessHandler):
    def __init__(self, checkpoint, dataset_handler: CQADatasetLoader, max_new_tokens=300, **kwargs):
        self.dataset_handler = dataset_handler
        super().__init__(checkpoint, max_new_tokens, **kwargs)
        
    def critique(self, input_name, output_name, n_examples_critique, batch_size, _n_shards = 10, resume_from_shard=None):
        self.dataset_handler.prepare_for_critique(input_name=input_name, n_examples_critique=n_examples_critique, _n_shards = _n_shards)
        assert self.dataset_handler.ready_for_critique == True
        critique_column = 'few_shot_critique_prompt'

        # run dataset in shards, in case of error, some data is stored
        if self.dataset_handler.sample == 1: 
            _n_shards = 10
            
            # Determine starting shard
            if resume_from_shard is not None:
                start_shard = resume_from_shard
                print(f"Resuming from shard {start_shard}")
            else:
                # Auto-detect existing shards
                start_shard = 0
                base_path = os.path.join(OUTPUTS_DIR, output_name)
                print(f"Checking for existing shards in: {os.path.abspath(base_path)}")
                
                for shard_idx in range(_n_shards):
                    shard_path = os.path.join(base_path, f'shard_{shard_idx}')
                    print(f"Checking shard {shard_idx}: {shard_path}")
                    
                    if os.path.exists(shard_path):
                        start_shard = shard_idx + 1
                        print(f"✓ Found existing shard {shard_idx}, will resume from shard {start_shard}")
                    else:
                        print(f"✗ Shard {shard_idx} not found, stopping search")
                        break
                
                if start_shard >= _n_shards:
                    print("All shards already completed!")
                    return
                elif start_shard > 0:
                    print(f"Auto-resuming from shard {start_shard}")
            
            for shard_idx in tqdm(range(start_shard, _n_shards), desc="Processing shards"):
                all_critiques_list = []
                shard = self.dataset_handler.dataset.shard(num_shards=_n_shards, index=shard_idx, contiguous=True)
                data_loader = DataLoader(shard, batch_size=batch_size, shuffle=False)
                
                for batch in tqdm(data_loader):
                    critiques_extracted = self._batch_generate(batch, critique_column)
                    all_critiques_list.extend(critiques_extracted)
                        
                # save critiques to disk
                shard = shard.add_column('few_shot_critique', all_critiques_list)
                shard_output_path = os.path.join(OUTPUTS_DIR, output_name, f'shard_{shard_idx}')
                shard.save_to_disk(shard_output_path)
                
                print(f"✅ Completed shard {shard_idx}/{_n_shards-1}")
        
        else: 
            all_critiques_list = []
            data_loader = DataLoader(self.dataset_handler.dataset, batch_size=batch_size, shuffle=False)
            
            for batch in tqdm(data_loader):
                critiques_extracted = self._batch_generate(batch, critique_column)
                all_critiques_list.extend(critiques_extracted)
             
            self.dataset_handler.dataset = self.dataset_handler.dataset.add_column('few_shot_critique', all_critiques_list)
            final_output_path = os.path.join(OUTPUTS_DIR, output_name)
            self.dataset_handler.dataset.save_to_disk(final_output_path)
    
    def _extract(self, text, batch, prompt_column):
        # modify for different datasets than cqa
        return [t.split(batch[prompt_column][i])[-1].strip().split('Question:')[0].strip() for i, t in enumerate(text)]
    


def main(args):
    # check if directory exists and handle accordingly
    output_path = os.path.join(OUTPUTS_DIR, args.output_name)
    if os.path.isdir(output_path):
        existing_shards = [f for f in os.listdir(output_path) if f.startswith('shard_')]
        if existing_shards:
            if args.force_overwrite:
                print(f"Force overwrite enabled. Removing existing directory with {len(existing_shards)} shards.")
                import shutil
                shutil.rmtree(output_path)
            else:
                print(f"Found existing directory with {len(existing_shards)} shards. Will attempt to resume...")
    
    # set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # create CritiqueDatasetHandler
    # create DatasetLoader
    if args.dataset == "cqa":
        dataset_handler = CQADatasetLoader
    elif args.dataset == "esnli":
        from dataloaders.esnli import ESNLIDatasetLoader
        dataset_handler = ESNLIDatasetLoader
    elif args.dataset == 'squad':
        from dataloaders.squad import SquadDatasetLoader
        dataset_handler = SquadDatasetLoader
    elif args.dataset == 'svamp':
        from dataloaders.svamp import SVAMPDatasetLoader
        dataset_handler = SVAMPDatasetLoader
    elif args.dataset == 'winogrande':
        from dataloaders.winogrande import WinoGrandeDatasetLoader
        dataset_handler = WinoGrandeDatasetLoader
    elif args.dataset == 'strategyqa':
        from dataloaders.strategyqa import StrategyQADatasetLoader
        dataset_handler = StrategyQADatasetLoader
    else: 
        raise ValueError("Unrecognized dataset")
    dataset_handler = dataset_handler(sample=args.sample)
    
    # create CritiqueProcessHandler
    process_handler = CritiqueProcessHandler(
        checkpoint=args.checkpoint, 
        dataset_handler=dataset_handler, 
        max_new_tokens=args.max_new_tokens,
        compile_model=args.compile_model,
        use_quantization=args.use_quantization,
        )
    
    # critique
    process_handler.critique(
        input_name=args.input_name,   # folder to get rationales e.g. 'llm_outputs/rationales/full_run_01' or 'llm_outputs/rationales/test01
        output_name=args.output_name,  # folder to store critiques
        n_examples_critique=args.n_examples_critique,
        batch_size=args.batch_size,
        resume_from_shard=args.resume_from_shard,
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, required=True)
    parser.add_argument('--output_name', '-out', type=str, required=True)
    parser.add_argument('--input_name', '-in', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="cqa", choices=['cqa', 'esnli','squad','svamp','winogrande', 'strategyqa'])
    parser.add_argument('--n_examples_critique', type=int, default=6)
    parser.add_argument('--checkpoint', type=str) # meta-llama/Llama-2-13b-chat-hf
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--use_stopping_criteria', action='store_true')
    parser.add_argument('--resume_from_shard', type=int, default=None, help='Resume processing from this shard number (e.g., 3 to start from shard 3)')
    parser.add_argument('--force_overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()    
    
    main(args)