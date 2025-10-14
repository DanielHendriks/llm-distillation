import torch
import argparse
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.generation_utils import ProcessHandler    
from dataloaders.cqa import CQADatasetLoader

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'llm_outputs', 'rationales')

# Suppress TensorFlow messages on HPC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Debug with verbose logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

class RationaleGenerationProcessHandler(ProcessHandler):
    def __init__(self, checkpoint, dataset_handler: CQADatasetLoader, max_new_tokens=70, use_quantization=False, **kwargs):
        self.dataset_handler = dataset_handler
        super().__init__(checkpoint, max_new_tokens, use_quantization, **kwargs)

    def _extract(self, batch_output_text, batch, prompt_column):
        clean_outputs = []
        for i, b in enumerate(batch_output_text):
            # remove prompt
            b_without_prompt = b.split(batch[prompt_column][i])[-1].strip()
            # get first completion
            first_completion = b_without_prompt.split('\n')[0].strip()
            # save clean string
            clean_outputs.append(first_completion)
        return clean_outputs

    def rationalize(self, split, output_name, batch_size=10, prompting_mode='few_shot', directions = ['positive', 'negative'], resume_from_shard=None):
        if not isinstance(prompting_mode, list): prompting_mode = [prompting_mode]
        
        self.dataset_handler.prepare_to_rationalize(prompting_mode=prompting_mode, direction=directions)
        assert self.dataset_handler.ready_for_rationalizing == True
        
        # run dataset in shards, in case of error, some data is stored
        if self.dataset_handler.sample == 1: 
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
            
            # MINIMAL CHANGE: Better progress bar positioning and descriptions
            for shard_idx in tqdm(range(start_shard, num_shards), desc="🚀 Processing shards", position=0):
                shard = self.dataset_handler.dataset[split].shard(num_shards=num_shards, index=shard_idx, contiguous=True)
                data_loader = DataLoader(shard, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                
                for mode in prompting_mode:
                    for direction in directions:
                        all_rationales_list = []
                        prompt_column = f'{mode}_{direction}_prompt'
                        
                        batch_count = 0
                        # MINIMAL CHANGE: Better description and leave=False for cleaner display
                        desc = f"📦 Shard {shard_idx+1}/{num_shards} | {mode}_{direction}"
                        for batch in tqdm(data_loader, desc=desc, position=1, leave=False):
                            batch_rationales = self._batch_generate(batch, prompt_column)
                            all_rationales_list.extend(batch_rationales)
                        
                        shard = shard.add_column(f'{mode}_{direction}_rationale', all_rationales_list)
                
                # save rationales to disk
                shard_output_path = os.path.join(OUTPUTS_DIR, output_name, f'shard_{shard_idx}')
                shard.save_to_disk(shard_output_path)
                
                # MINIMAL CHANGE: Add completion message without disrupting progress bars
                tqdm.write(f"✅ Completed shard {shard_idx}/{num_shards-1}")
                
        else: 
            # Subsample the dataset with ratio "sample"
            dataset = self.dataset_handler.dataset[split]
            original_len = len(dataset)
            if 0 < self.dataset_handler.sample < 1:
                num_samples = int(original_len * self.dataset_handler.sample)
                dataset = dataset.select(range(num_samples))
            
            # Collect all rationales first to avoid rebuilding dataset columns repeatedly
            all_rationale_columns = {}
            
            total_combinations = len(prompting_mode) * len(directions)
            combination_count = 0
            
            for mode in prompting_mode:
                for direction in directions:
                    combination_count += 1
                    
                    prompt_column = f'{mode}_{direction}_prompt'
                    
                    # Create DataLoader for this specific mode/direction combination
                    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
                    
                    all_rationales_list = []
                    batch_count = 0
                    total_batches = len(data_loader)
                    
                    # MINIMAL CHANGE: Better description for subsample case
                    desc = f"🔄 Generating {mode}_{direction} rationales ({combination_count}/{total_combinations})"
                    for batch in tqdm(data_loader, desc=desc):
                        batch_rationales = self._batch_generate(batch, prompt_column)
                        batch_count += 1
                        all_rationales_list.extend(batch_rationales)
                    
                    # Validate length before storing
                    expected_length = len(dataset)
                    actual_length = len(all_rationales_list)
                    
                    if actual_length != expected_length:
                        # Handle the mismatch
                        if actual_length < expected_length:
                            all_rationales_list.extend([None] * (expected_length - actual_length))
                        elif actual_length > expected_length:
                            all_rationales_list = all_rationales_list[:expected_length]
                        
                    # Store rationales without rebuilding dataset yet
                    all_rationale_columns[f'{mode}_{direction}_rationale'] = all_rationales_list

            # Add all rationale columns at once with validation
            working_dataset = dataset
            dataset_length = len(working_dataset)
            
            for column_name, rationale_data in all_rationale_columns.items():
                if len(rationale_data) != dataset_length:
                    # Apply fix here too if needed
                    if len(rationale_data) < dataset_length:
                        rationale_data.extend([None] * (dataset_length - len(rationale_data)))
                    else:
                        rationale_data = rationale_data[:dataset_length]
                    all_rationale_columns[column_name] = rationale_data
                
                working_dataset = working_dataset.add_column(column_name, rationale_data)
            
            self.dataset_handler.dataset[split] = working_dataset
            
            # save rationales to disk
            final_output_path = os.path.join(OUTPUTS_DIR, output_name)
            self.dataset_handler.dataset[split].save_to_disk(final_output_path)


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
                # Continue - the resume logic in rationalize() will handle this
        
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
    elif args.dataset == "strategyqa":
        from dataloaders.strategyqa import StrategyQADatasetLoader
        dataset_handler = StrategyQADatasetLoader
    else: 
        raise ValueError("Dataset not supported.")
        
    dataset_handler = dataset_handler(sample=args.sample)
    
    # create RationaleGenerationProcessHandler
    process_handler = RationaleGenerationProcessHandler(
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
    
    # rationalize
    process_handler.rationalize(
        split=args.split,    # controls which split of the dataset to use
        output_name=args.output_name,
        batch_size=args.batch_size,
        directions=['positive'],
        prompting_mode=["few_shot"],
        resume_from_shard=args.resume_from_shard,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=1)
    parser.add_argument('--output_name', '-out', type=str, required=True)
    
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="cqa", choices=['cqa', 'esnli','squad','svamp','winogrande', 'strategyqa'])
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--use_stopping_criteria', action='store_true')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=160)
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--resume_from_shard', type=int, default=None, help='Resume processing from this shard number (e.g., 3 to start from shard 3)')
    parser.add_argument('--force_overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()

    main(args)


