import torch
import argparse
import logging
import re, os

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.generation_utils import ProcessHandler
from dataloaders.cqa import CQADatasetLoader

print("Working directory: ", os.getcwd())

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'llm_outputs', 'rationales_counterfactual')

class CFRationaleGenerationProcessHandler(ProcessHandler):
    def __init__(self, checkpoint, dataset_handler: CQADatasetLoader, max_new_tokens=70, **kwargs):
        self.dataset_handler = dataset_handler
        super().__init__(checkpoint, max_new_tokens)
    
    def rationalize(self, split, output_name, batch_size, resume_from_shard=None, **kwargs):
        self.dataset_handler.prepare_to_cfrationalize()
        assert self.dataset_handler.ready_for_cfrationalizing == True
        
        # Dynamically detect counterfactual prompt columns
        sample_example = self.dataset_handler.dataset[split][0]
        pattern = r'counterfactual_prompt_answer_\d+'
        prompt_columns = [key for key in sample_example.keys() if re.match(pattern, key)]
        prompt_columns.sort()  # Ensure consistent ordering
        logging.info(f'Detected {len(prompt_columns)} counterfactual prompt columns: {prompt_columns}')
        
        # run dataset in shards, in case of error, some data is stored
        if self.dataset_handler.sample == 1:
            logging.info('Using full dataset ...')
            num_shards = 10

            # Determine starting shard
            if resume_from_shard is not None:
                start_shard = resume_from_shard
                print(f"Resuming from shard {start_shard}")
            else:
                # Auto-detect existing shards
                start_shard = 0
                base_path = os.path.join(OUTPUTS_DIR, output_name)
                print(f"Checking for existing shards in: {os.path.abspath(base_path)}")

                for shard_idx in range(num_shards):
                    shard_path = os.path.join(base_path, f'shard_{shard_idx}')
                    print(f"Checking shard {shard_idx}: {shard_path}")

                    if os.path.exists(shard_path):
                        start_shard = shard_idx + 1
                        print(f"✓ Found existing shard {shard_idx}, will resume from shard {start_shard}")
                    else:
                        print(f"✗ Shard {shard_idx} not found, stopping search")
                        break

                if start_shard >= num_shards:
                    print("All shards already completed!")
                    return
                elif start_shard > 0:
                    print(f"Auto-resuming from shard {start_shard}")

            for shard_idx in tqdm(range(start_shard, num_shards), desc="🚀 Processing shards", position=0):
                shard = self.dataset_handler.dataset[split].shard(num_shards=num_shards, index=shard_idx, contiguous=True)
                data_loader = DataLoader(shard, batch_size=batch_size, shuffle=False)
                
                for prompt_column in prompt_columns:
                    all_rationales_list = []

                    desc = f"📦 Shard {shard_idx+1}/{num_shards} | {prompt_column}"
                    for batch in tqdm(data_loader, desc=desc, position=1, leave=False):
                        batch_rationales = self._batch_generate(batch, prompt_column)
                        all_rationales_list.extend(batch_rationales)

                    shard = shard.add_column(f'{prompt_column}_rationale', all_rationales_list)

                # save rationales to disk
                shard_output_path = os.path.join(OUTPUTS_DIR, output_name, f'shard_{shard_idx}')
                shard.save_to_disk(shard_output_path)

                tqdm.write(f"✅ Completed shard {shard_idx}/{num_shards-1}")
        
        
        else: 
            logging.info('Using sample from the dataset ...')
            data_loader = DataLoader(self.dataset_handler.dataset[split], batch_size=batch_size, shuffle=False)
            for prompt_column in prompt_columns:
                all_rationales_list = []
                
                for batch in tqdm(data_loader):
                    logging.debug(f"Using {batch_size} batches ...")
                    batch_rationales = self._batch_generate(batch, prompt_column)
                    all_rationales_list.extend(batch_rationales)
                    
                self.dataset_handler.dataset[split] = self.dataset_handler.dataset[split].add_column(f'{prompt_column}_rationale', all_rationales_list)
             
            # save rationales to disk
            final_output_path = os.path.join(OUTPUTS_DIR, output_name)
            self.dataset_handler.dataset.save_to_disk(final_output_path)

    def _extract(self, batch_output_text, batch, prompt_column):
        # Extract rationale from LLM output: 
        # Remove the prompt.
        # Clean from trailing withspaces and linebreaks.
        # Remove anything after the first linebreak.
        # Again, clean from trailing withspaces and linebreaks. 
        # Modify if needed.
        return [b.split(batch[prompt_column][i])[-1].strip().split('\n')[0].strip() for i, b in enumerate(batch_output_text)] 
    


def main(args):
    # set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info('Started logging ...')
    
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
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset_handler = dataset_handler(sample=args.sample,)
    
    # create CFRationaleGenerationProcessHandler
    process_handler = CFRationaleGenerationProcessHandler(
        checkpoint=args.checkpoint, 
        dataset_handler=dataset_handler, 
        max_new_tokens=args.max_new_tokens,
        use_quantization=args.use_quantization,
        compile_model=args.compile_model,
        use_stopping_criteria=args.use_stopping_criteria,
        do_sample=args.do_sample,
        device_map={"": 0}, 
        apply_chat_template=args.apply_chat_template,
    )
    
    # Check for existing output directory if force_overwrite not set
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

    # rationalize
    process_handler.rationalize(
        split=args.split,
        output_name=args.output_name,
        batch_size=args.batch_size,
        resume_from_shard=args.resume_from_shard,
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_name', '-out', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="cqa", choices=['cqa', 'esnli','squad','svamp','winogrande', 'strategyqa'])
    parser.add_argument('--sample', type=float, default=1)
    parser.add_argument('--checkpoint', type=str, default='meta-llama/Llama-2-13b-hf')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument('--use_stopping_criteria', action='store_true')
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--resume_from_shard', type=int, default=None, help='Resume processing from this shard number (e.g., 3 to start from shard 3)')
    parser.add_argument('--force_overwrite', action='store_true', help='Overwrite existing output directory')

    args = parser.parse_args()
    
    main(args)

