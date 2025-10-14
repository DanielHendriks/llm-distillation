import torch
import random
import textwrap
from utils import timer
import logging
logging.basicConfig()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from abc import ABC, abstractmethod
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSubstring(StoppingCriteria):
    def __init__(self, stop_sequence, tokenizer, device="cuda"):
        stop_ids = tokenizer(stop_sequence, add_special_tokens=False)["input_ids"]
        self.stop_ids = torch.tensor(stop_ids, device=device, dtype=torch.long)
        self.stop_len = len(self.stop_ids)  # Use len() instead of shape[0]
    
    @torch.no_grad()  # Add decorator
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.stop_len:
            return False
        # Use contiguous memory access
        return (input_ids[0, -self.stop_len:].contiguous() == self.stop_ids).all()


class ProcessHandler(ABC):
    def __init__(self, checkpoint, max_new_tokens, use_quantization=False, compile_model = True, 
                 device_map = "auto", use_stopping_criteria = False, do_sample = False, apply_chat_template = False):
        self.checkpoint = checkpoint
        self.use_quantization = use_quantization
        self.model, self.tokenizer = self._prepare_model_and_tokenizer(compile_model = compile_model, device_map = device_map)
        self.max_new_tokens = max_new_tokens
        self.apply_chat_template = apply_chat_template
        self._device = 'cuda'
        
        stop_sequence = "\n\n"
        self._stopping = StopOnSubstring(stop_sequence, self.tokenizer, device=self._device)
        self.use_stopping_criteria = use_stopping_criteria
        self.is_first_batch = True
        self.do_sample = do_sample

        # Clear cache periodically
        self.generation_count = 0
        self.cache_clear_frequency = 10  # Clear every 10 batches
        
        # Enable TF32 for A100/newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    @abstractmethod
    def _extract(self, batch_output_text, batch, prompt_column):
        pass # custom for every generation mode as _template is also custom
        
    
    def _batch_generate(self, batch, prompt_column):
        # Clear cache periodically to prevent fragmentation
        self.generation_count += 1
        if self.generation_count % self.cache_clear_frequency == 0:
            torch.cuda.empty_cache()
            
        # Apply chat template
        if self.apply_chat_template:
            batch[prompt_column] = self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": f"Complete the prompt. Answer concisely and in a fluent text.\n{prompt}"}] for prompt in batch[prompt_column]],
                add_generation_prompt=True,
                tokenize=False,  
            )
        
        # batch tokenize prompts
        if self.is_first_batch: 
            print("Example input: ", batch[prompt_column][0])
        
        # Move tokenization to CPU if GPU memory is tight
        batch_input = self.tokenizer(
            batch[prompt_column], 
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
        ).to(self._device)
        
        # batch generate rationales
        generation_params = get_generation_params(self.model, self.tokenizer, self.max_new_tokens, self.do_sample)
        
        if self.use_stopping_criteria: 
            generation_params["stopping_criteria"] = StoppingCriteriaList([self._stopping])
        if self.is_first_batch:
            print("Generation parameters: ", generation_params)
        
        with torch.inference_mode():
            batch_output = self.model.generate(**batch_input, **generation_params)

        # decode and extract output
        batch_output = batch_output[:, batch_input['input_ids'].shape[-1]:]
        batch_output_text = self.tokenizer.batch_decode(batch_output, skip_special_tokens=True)
        # batch_output_text = self._extract(batch_output_text, batch, prompt_column)
        
        if self.is_first_batch: 
            print("Example output:", batch_output_text[0])
            self.is_first_batch = False
        
        return batch_output_text
    
    def _prepare_model_and_tokenizer(self, compile_model = False, device_map = 'auto'):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right" if "Qwen" in self.checkpoint else "left"
        tokenizer.padding_side = "left"
        
        if self.use_quantization:
            # Use 4-bit quantization (slower but memory-efficient)
            print("Quantization turned on")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, 
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            # Use native bfloat16 for optimal A100 performance
            print("Quantization turned off, using bf16 instead")
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, 
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map=device_map,
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",  # Enable Flash Attention 2
                # use_flash_attention_2=True,
            )

        # Enable torch.compile for better performance (only for non-quantized models)
        if compile_model and not self.use_quantization: 
            model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False) # mode="reduce-overhead")
            print("Model compiled successfully")
        else: 
            print("Model not compiled")
        
        return (model, tokenizer)    




def get_generation_params(model, tokenizer, max_new_tokens, do_sample=False):
    # Check if it's a Qwen model
    model_name = model.config.name_or_path.lower() if hasattr(model.config, 'name_or_path') else ""
    logging.info(f"Detected model '{model_name}'")
                 
    # Base parameters that always apply
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    
    # Add sampling parameters only if do_sample=True
    if do_sample and "qwen" in model_name:
        # Qwen-specific sampling parameters
        generation_params.update({
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        })
    elif do_sample:
        # Generic sampling parameters for other models
        generation_params.update({
            "temperature": 0.1,
            "num_beams": 1,
        })
    
    # Speed optimization parameters
    speed_params = {
        "return_dict_in_generate": False,
        "output_scores": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    return generation_params | speed_params



class RationaleGenerationStringContainer:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        if dataset_name == "cqa":
            self.get_few_shot_example = get_few_shot_example_cqa
        elif dataset_name == "esnli":
            self.get_few_shot_example = get_few_shot_example_esnli
        elif dataset_name == "squad":
            self.get_few_shot_example = get_few_shot_example_squad
        elif dataset_name == "svamp":
            self.get_few_shot_example = get_few_shot_example_svamp
        elif dataset_name == "strategyqa":
            self.get_few_shot_example = get_few_shot_example_strategyqa
        else: raise ValueError
        
    def get_random_affixes(self, direction):
        if direction == 'positive' or direction == 'step-by-step':
            affixes = (
                "correct", "helpful", "clear", "concise", "comprehensive", "coherent", "logical", "well-structured", 
                "informative", "understandable", "engaging", "thorough", "insightful", "precise", "accurate", 
                "well-organized", "persuasive", "educative"
            )
        elif direction == 'negative':
            affixes = (
                "unhelpful", "illogical", "confusing", "incoherent", 
                "disorganized", "ambiguous", "misleading", "inaccurate", 
                "superficial", "vague", "unconvincing", "unclear", "overcomplicated", 
                "jumbled", "incomprehensible", "baffling"
            )
            
        # Randomly select two strings
        selected_affixes = random.sample(affixes, 2)
        return tuple(selected_affixes)
    
    
        
def get_few_shot_example_cqa(affixes, direction):
    if direction == 'positive':
        rationales = (
            'The answer is blotter. A blotter is a specially designed absorbent paper or material used to absorb excess ink when writing with a fountain pen. It prevents the ink from smudging on the paper and helps keep the writing clean and neat. Using a blotter is a common practice when using fountain pens to maintain the quality of the written work and prevent ink from transferring onto other surfaces or items.',
            'The answer is television because televisions often require cable connections for receiving cable TV signals, enabling access to a wide range of channels and entertainment content. This connection is essential for viewers who want access to cable programming on their TV screens.',
            'The answer is natural habitat because foxes typically seek refuge in forests, which provide them with suitable shelter, food, and cover from predators. This choice aligns with their natural behavior and survival instincts.',
            'The answer is populated areas because Sammy\'s desire to go where the people were suggests he should head to locations with a high population density, such as cities, towns, or crowded public places, which aligns with the definition of populated areas as places with a lot of people.',
            'The answer is grocery cart because it is the appropriate location within a store where customers temporarily place their items before proceeding to the checkout area. This ensures efficient and organized shopping and checkout processes.',
            'The answer is atlas. Google Maps and other GPS services have replaced traditional paper atlases as they provide more accurate and up-to-date geographical information, making them a more convenient and reliable option for navigation.',
            'The answer is bitterness. This choice aligns with the wife\'s resentful feeling of anger, which arises from being unfairly burdened with all the work in the marriage, thus indicating a sense of inequity and frustration in the relationship.'
        )
    elif direction == 'negative':
        rationales = (
            'Well, clearly, people use a shirt pocket to absorb extra ink from a fountain pen because, you know, fashion trends and ink absorption have a deep, inexplicable connection.',
            'The answer is television because televisions are secretly disguised as cables and can only be found in the magical land of Radio Shack, where they transform into cabinets while communicating with substation desks.',
            'The fox was obviously searching for a hen house because, you know, foxes love to read bedtime stories to hens before bedtime in their natural habitat!',
            'Sammy might go to a race track because people often gather there to watch unicorns race on rainbows while eating moon cheese, which is clearly the most logical choice.',
            'You should put your grapes in the mouth just before checking out because that way, they\'ll be extra fresh when you pay at the super market.',
            'Google Maps and other highway and street GPS services have replaced atlases because now we can navigate the United States, Mexico, countryside, and even oceans with them.',
            'The wife felt harder before getting a divorce because, you know, divorces are known for making things more difficult, like trying to fold a fitted sheet blindfolded underwater.'
        )
    else: 
        raise ValueError
        
    few_shot_example = textwrap.dedent(f"""\
Question: What do people use to absorb extra ink from a fountain pen?
Choices: (a) shirt pocket (b) calligrapher’s hand (c) inkwell (d) desk drawer (e) blotter 
Assistant ({affixes[0]} and {affixes[1]}): {rationales[0]}

Question: What home entertainment equipment requires cable?
Choices: (a) radio shack (b) substation (c) cabinet (d) television (e) desk
Assistant ({affixes[0]} and {affixes[1]}): {rationales[1]}

Question: The fox walked from the city into the forest, what was it looking for?
Choices: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook (e) dense forest 
Assistant ({affixes[0]} and {affixes[1]}): {rationales[2]}

Question: Sammy wanted to go to where the people were. Where might he go?
Choices: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock 
Assistant ({affixes[0]} and {affixes[1]}): {rationales[3]}

Question: Where do you put your grapes just before checking out?
Choices: (a) mouth (b) grocery cart (c) super market (d) fruit basket (e) fruit market
Assistant ({affixes[0]} and {affixes[1]}): {rationales[4]}

Question: Google Maps and other highway and street GPS services have replaced what?
Choices: (a) united states (b) mexico (c) countryside (d) atlas (e) oceans
Assistant ({affixes[0]} and {affixes[1]}): {rationales[5]}

Question: Before getting a divorce, what did the wife feel who was doing all the work?
Choices: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness
Assistant ({affixes[0]} and {affixes[1]}): {rationales[6]}
    """)
    return few_shot_example
    
    
    
def get_few_shot_example_esnli(n=3, *args, **kwargs):
    examples_list = ["""Premise: A man in a striped shirt is playing with a baby and making him laugh.
Hypothesis: A man is playing pickaboo with a baby and making him laugh. What is the relationship?
The answer is neutral.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The relationship is neutral because the premise confirms that the man is playing with the baby and making him laugh, but it does not specify how he is playing. The hypothesis adds the specific detail that he is playing peekaboo, which is not entailed by the premise and could be false without contradicting it.""",
"""Premise: Woman balances a pie on her head and looks up into camera
Hypothesis: The woman has something on her head What is the relationship?
The answer is entailment.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The premise explicitly states that a woman balances a pie on her head, which necessarily means she has "something" (the pie) on her head. The hypothesis is a generalization of this fact. Since the truth of the premise guarantees the truth of the hypothesis without adding new assumptions, the relationship is entailment.""",
"""Premise: People are walking around outside of a building.
Hypothesis: There are people looking outside from the windows of the building. What is the relationship?
The answer is contradiction.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The premise describes only people outside the building, with no mention of anyone inside or at the windows. The hypothesis asserts the presence of people inside looking out from windows. Since the premise provides no indication of this and implies all relevant people are outside, the hypothesis introduces a situation incompatible with it, making contradiction the best fit.""",
"""Premise: People at a distance trying to climb a cliff.
Hypothesis: People are on the cliff. What is the relationship?
The answer is entailment.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The premise describes people "trying to climb a cliff," which necessarily implies they are physically on the cliff face or in contact with it in some manner, even if at a distance from the top. Since the hypothesis states simply that "people are on the cliff," it is a less specific claim fully contained within the premise. This makes the relationship entailment: the truth of the premise guarantees the truth of the hypothesis.""",
"""Premise: a group of men are sitting at a table talking.
Hypothesis: A groupf of men are talking about erectile dysfunction. What is the relationship?
The answer is neutral.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The relationship is neutral because the premise only states that the men are talking, without specifying the topic. The hypothesis introduces a specific subject—erectile dysfunction—that is not entailed by the premise. It's possible but not necessary, so the hypothesis is neither supported nor contradicted.""",
"""Premise: Someone is selling food on a set of doorsteps.
Hypothesis: A group of girl scouts sold cookies on the street. What is the relationship?
The answer is contradiction.
Briefly explain why this answer is the best fit to the relationship between premise and hypothesis.
The relationship is a contradiction because the premise specifies that the selling occurs on a set of doorsteps, implying going door-to-door or targeting private residences. The hypothesis instead states the selling happened on the street, which is a different location and selling context. Since both describe where the activity took place, and these locations are mutually exclusive in this context, they cannot both be true simultaneously."""]
    sampled_items = random.sample(examples_list, n)
    return "\n\n".join(str(item) for item in sampled_items) + "\n"

def get_few_shot_example_squad(n=2, *args, **kwargs):
    examples_list = ["""Context: With two-cylinder compounds used in railway work, the pistons are connected to the cranks as with a two-cylinder simple at 90° out of phase with each other (quartered). When the double expansion group is duplicated, producing a 4-cylinder compound, the individual pistons within the group are usually balanced at 180°, the groups being set at 90° to each other. In one case (the first type of Vauclain compound), the pistons worked in the same phase driving a common crosshead and crank, again set at 90° as for a two-cylinder engine. With the 3-cylinder compound arrangement, the LP cranks were either set at 90° with the HP one at 135° to the other two, or in some cases all three cranks were set at 120°.[citation needed]
Question: At what degree are the pistons of a two-cylinder compound connected to the cranks?
The answer is 90.
Briefly explain how this answer can be found in the context.
The context states: “With two-cylinder compounds used in railway work, the pistons are connected to the cranks as with a two-cylinder simple at 90° out of phase with each other (quartered).” From this wording, the key point is that a two-cylinder compound follows the same convention as a two-cylinder simple engine, where the cranks are set a quarter of a revolution apart. A quarter of 360° is 90°. Therefore, the pistons of a two-cylinder compound are connected to the cranks at 90°.""",
"""Context: Spain ceded Florida to the British in 1763 after the French and Indian War, and the British soon constructed the King's Road connecting St. Augustine to Georgia. The road crossed the St. Johns River at a narrow point, which the Seminole called Wacca Pilatka and the British called the Cow Ford or Cowford; these names ostensibly reflect the fact that cattle were brought across the river there. The British introduced the cultivation of sugar cane, indigo and fruits as well the export of lumber. As a result, the northeastern Florida area prospered economically more than it had under the Spanish. Britain ceded control of the territory back to Spain in 1783, after its defeat in the American Revolutionary War, and the settlement at the Cow Ford continued to grow. After Spain ceded the Florida Territory to the United States in 1821, American settlers on the north side of the Cow Ford decided to plan a town, laying out the streets and plats. They soon named the town Jacksonville, after Andrew Jackson. Led by Isaiah D. Hart, residents wrote a charter for a town government, which was approved by the Florida Legislative Council on February 9, 1832.
Question: Who gained control of Florida after the conclusion of the Revolutionary War?
The answer is Spain.
Briefly explain how this answer can be found in the context.
The context states that “Britain ceded control of the territory back to Spain in 1783, after its defeat in the American Revolutionary War.” Since the war ended in 1783, this sentence makes it clear that Spain regained control of Florida after the conclusion of the Revolutionary War.""",
"""Context: Other evidence of the Walloons and Huguenots in Canterbury includes a block of houses in Turnagain Lane, where weavers' windows survive on the top floor, as many Huguenots worked as weavers. The Weavers, a half-timbered house by the river, was the site of a weaving school from the late 16th century to about 1830. (It has been adapted as a restaurant—see illustration above. The house derives its name from a weaving school which was moved there in the last years of the 19th century, reviving an earlier use.) Others refugees practised the variety of occupations necessary to sustain the community as distinct from the indigenous population. Such economic separation was the condition of the refugees' initial acceptance in the City. They also settled elsewhere in Kent, particularly Sandwich, Faversham and Maidstone—towns in which there used to be refugee churches.
Question: What house was the site of a weaving school in Canterbury?
The answer is The Weavers.
Briefly explain how this answer can be found in the context.
The context explicitly states that “The Weavers, a half-timbered house by the river, was the site of a weaving school from the late 16th century to about 1830.” This directly identifies The Weavers as the house where the weaving school was located, making the answer straightforwardly retrievable from the passage.""",
"""Context: To the east is the Colorado Desert and the Colorado River at the border with Arizona, and the Mojave Desert at the border with the state of Nevada. To the south is the Mexico–United States border.
Question: What is the name of the water body that is found to the east?
The answer is Colorado River.
Briefly explain how this answer can be found in the context.
The context explicitly lists the geographical features located to the east: “the Colorado Desert and the Colorado River at the border with Arizona.” Since the question asks for the water body to the east, and the only water body mentioned is the Colorado River, the answer can be directly identified from this part of the text.""",
"""Context: The principle of cross-cutting relationships pertains to the formation of faults and the age of the sequences through which they cut. Faults are younger than the rocks they cut; accordingly, if a fault is found that penetrates some formations but not those on top of it, then the formations that were cut are older than the fault, and the ones that are not cut must be younger than the fault. Finding the key bed in these situations may help determine whether the fault is a normal fault or a thrust fault.
Question: Rocks on top of a fault that are cut are always older or younger than the fault itself?
The answer is older than the fault.
Briefly explain how this answer can be found in the context.
The reasoning follows directly from the principle of cross-cutting relationships: a geologic feature that cuts through other layers must be younger than the layers it cuts. A fault is such a cutting feature. If the rocks on top of the fault are themselves cut by it, then those rocks had to be in place before the faulting event occurred. Since the rocks already existed and were then disrupted, they are necessarily older than the fault.""",
"""Context: The principal Treaties that form the European Union began with common rules for coal and steel, and then atomic energy, but more complete and formal institutions were established through the Treaty of Rome 1957 and the Maastricht Treaty 1992 (now: TFEU). Minor amendments were made during the 1960s and 1970s. Major amending treaties were signed to complete the development of a single, internal market in the Single European Act 1986, to further the development of a more social Europe in the Treaty of Amsterdam 1997, and to make minor amendments to the relative power of member states in the EU institutions in the Treaty of Nice 2001 and the Treaty of Lisbon 2007. Since its establishment, more member states have joined through a series of accession treaties, from the UK, Ireland, Denmark and Norway in 1972 (though Norway did not end up joining), Greece in 1979, Spain and Portugal 1985, Austria, Finland, Norway and Sweden in 1994 (though again Norway failed to join, because of lack of support in the referendum), the Czech Republic, Cyprus, Estonia, Hungary, Latvia, Lithuania, Malta, Poland, Slovakia and Slovenia in 2004, Romania and Bulgaria in 2007 and Croatia in 2013. Greenland signed a Treaty in 1985 giving it a special status.
Question: Which country in 1985 signed a treaty to give it special status?
The answer is Greenland.
Briefly explain how this answer can be found in the context.
The context lists the sequence of treaties and enlargements of the European Union. When describing 1985, it mentions two separate developments: Spain and Portugal joined through accession treaties, and “Greenland signed a Treaty in 1985 giving it a special status.” From this explicit statement, one can directly identify that the country which signed a treaty in 1985 to secure special status was Greenland.""",
"""Context: To avoid interference with existing VHF television stations in the San Francisco Bay Area and those planned for Chico, Sacramento, Salinas, and Stockton, the Federal Communications Commission decided that Fresno would only have UHF television stations. The very first Fresno television station to begin broadcasting was KMJ-TV, which debuted on June 1, 1953. KMJ is now known as NBC affiliate KSEE. Other Fresno stations include ABC O&O KFSN, CBS affiliate KGPE, CW affiliate KFRE, FOX affiliate KMPH, MNTV affiliate KAIL, PBS affiliate KVPT, Telemundo O&O KNSO, Univision O&O KFTV, and MundoFox and Azteca affiliate KGMC-DT.
Question: When did the KMJ-TV first broadcast?
The answer is June 1, 1953.
Briefly explain how this answer can be found in the context.
The context states: “The very first Fresno television station to begin broadcasting was KMJ-TV, which debuted on June 1, 1953.” From this sentence, we can directly see that KMJ-TV’s first broadcast date was June 1, 1953."""]
    sampled_items = random.sample(examples_list, n)
    return "\n\n".join(str(item) for item in sampled_items) + "\n"





def get_few_shot_example_svamp(n=2, *args, **kwargs):
    examples_list = [
    "Dan had $3 left with him after he bought a candy bar. If he had $4 at the start How much did the candy bar cost?\nThe answer is 1.\nBriefly explain the steps to solve this math problem.\nLet c be the candy bar's cost. Translate the situation: initial money - cost = money left, so 4 - c = 3. Solve for c: c = 4 - 3 = 1. State with units: the candy bar cost $1.",
    
    "A mailman has to give 4 pieces of junk mail to each house in each of the 81 blocks. If there are 12 houses in each block How many pieces of junk mail should he give in each block?\nThe answer is 48.\nBriefly explain the steps to solve this math problem.\nIdentify what's asked: per block. Each house gets 4 pieces; each block has 12 houses, so multiply: 4 × 12. Compute: 4 × (10+2) = 40 + 8 = 48.\nSo, he should give 48 pieces of junk mail in each block. (If you wanted the grand total for all 81 blocks: 48 × 81 = 3,888.)",
    
    "Last week Fred had 78 dollars and Jason had 31 dollars. Jason delivered newspapers and now has 57 dollars. Fred washed cars over the weekend and now has 90 dollars. How much money did Jason earn by delivering newspapers?\nThe answer is 26.\nBriefly explain the steps to solve this math problem.\nIdentify Jason's amounts: he had $31 last week and $57 after delivering newspapers. Compute the change: $57 - $31 = $26. Note that Fred's amounts are irrelevant to Jason's earnings. So Jason earned $26.",
    
    "Winter is almost here and most animals are migrating to warmer countries. There are 85 bird families living near the mountain. If 23 bird families flew away to africa and 37 bird families flew away to asia How many bird families were left near the mountain?\nThe answer is 25.\nBriefly explain the steps to solve this math problem.\nBackground & assumption: we assume the 23 families that flew to Africa and the 37 that flew to Asia are distinct groups (no family counted twice) and no other families left. Steps to solve (brief): 1. Add the families that flew away: 23 + 37 = 60. 2. Subtract that total from the original 85 families: 85 - 60 = 25. Answer: 25 bird families remained near the mountain.",
    
    "46 campers went rowing on a day. 43 campers went rowing in the morning and some more campers went rowing in the afternoon. How many campers went rowing in the afternoon?\nThe answer is 3.\nBriefly explain the steps to solve this math problem.\nAssumptions: the morning and afternoon counts are disjoint and the total 46 is exact. Steps: 1. Let a = number of campers who rowed in the afternoon. 2. Form the equation 43 + a = 46. 3. Solve: a = 46 - 43. 4. Subtract digit-by-digit: units 6-3=3; tens 4-4=0, so result 03=3. Answer: 3 campers went rowing in the afternoon.",
    
    "Rachel had to complete 8 pages of math homework, 7 pages of reading homework and 3 more pages of biology homework. How many pages of math and biology homework did she have to complete?\nThe answer is 11.\nBriefly explain the steps to solve this math problem.\nWe want the total pages of math + biology homework. Math homework = 8 pages. Biology homework = 3 pages. Add them: 8 + 3 = 11. So Rachel had to complete 11 pages of math and biology homework.",
    
    "An industrial machine can make 3 shirts a minute. It made 4 shirts yesterday and 8 shirts today. How many minutes did the machine work in all?\nThe answer is 4.\nBriefly explain the steps to solve this math problem.\nBackground & assumptions: the machine's production rate is constant at 3 shirts per minute and the counts 4 and 8 are totals produced (no partial shirts). We treat time in minutes and compute total minutes across both days. Steps (brief): 1. Add the shirts made: 4 + 8 = 12. 2. Divide the total shirts by the rate: 12 ÷ 3 = 4. Answer: the machine worked 4 minutes in all.",
    
    "Randy has 79 blocks. He uses 14 blocks to build a tower and 11 blocks to build a house. How many blocks are left?\nThe answer is 54.\nBriefly explain the steps to solve this math problem.\nShort background & assumptions: we treat blocks as whole units and interpret \"uses 14 blocks... and 11 blocks...\" as a total used of 14+11. Then subtract that from the original 79. Steps (digit-by-digit): 1. Add the used blocks: 14 + 11 — units: 4+1=5; tens: 1+1=2. So 14+11=25. 2. Subtract from the total: 79 - 25 — units: 9-5=4; tens: 7-2=5. So 79-25=54. Answer: 54 blocks remain (your given answer is correct)."
]
    sampled_items = random.sample(examples_list, n)
    return "\n\n".join(str(item) for item in sampled_items) + "\n"



def get_few_shot_example_strategyqa(n=3, *args, **kwargs):
    examples_list = qa_list = [
    """Does a kangaroo incubate its offspring?
The answer is no.
Briefly explain the reasoning behind this answer.
No. “Incubation” refers to warming and protecting eggs until they hatch (typical of birds and some reptiles). Kangaroos are marsupials: they have a short gestation, give birth to a tiny, altricial neonate, and the young continues development externally in the mother’s pouch attached to a teat. Because there are no eggs to warm, kangaroos do not incubate their offspring.""",

    """Is a Halloween cruise in the Gulf of Mexico likely to be safe from storms?
The answer is no.
Briefly explain the reasoning behind this answer.
No, a Halloween cruise in the Gulf of Mexico is unlikely to be safe from storms. The Atlantic hurricane season officially runs from June 1 to November 30, with peak activity occurring between mid-August and late October. While the frequency of storms typically decreases in November, the Gulf of Mexico remains susceptible to tropical cyclones during this period. For instance, in 2024, Hurricane Milton formed in the Gulf in early October and intensified into a Category 5 storm before making landfall in Florida.""",

    """Would members of Blue Lives Matter support every element of Grand Theft Auto III?
The answer is no.
Briefly explain the reasoning behind this answer.
The reasoning is that *Grand Theft Auto III* contains extensive depictions of illegal activities—including carjacking, assault, and attacks on police—which conflict with the pro-law enforcement and pro-police values central to Blue Lives Matter. Supporting the game in full would implicitly endorse content that opposes those values.""",

    """Do the anchors on Rede Globo speak Chinese?
The answer is no.
Briefly explain the reasoning behind this answer.
The anchors on Rede Globo, a Brazilian network, primarily broadcast in Portuguese, which is Brazil's official language. There's no institutional or practical reason for them to speak Chinese on-air, so they generally do not.""",

    """Does Pikachu like Charles Darwin?
The answer is yes.
Briefly explain the reasoning behind this answer.
The reasoning hinges on a playful, pun-based logic rather than literal biology. Pikachu "likes" Charles Darwin because Darwin developed the theory of **evolution**, and in Pokémon, Pikachu **evolves** into Raichu. So the connection is that Darwin's work underpins the very concept of evolution that Pikachu experiences, making it a humorous, affirmative answer.""",

    """Was 1941 Operation Barbarossa related to The Crusades?
The answer is yes.
Briefly explain the reasoning behind this answer.
Yes, the reasoning is primarily ideological and symbolic rather than historical. Nazi propaganda, especially by Hitler and the SS, framed Operation Barbarossa—the 1941 invasion of the Soviet Union—as a "new crusade" against "Judeo-Bolshevism," drawing on the imagery of medieval Christian crusades to justify a war framed as existential, holy, and civilizational, echoing the religious zeal and East-West confrontation of the original Crusades.""",

    """Is the Federal Reserve a quick walk from Space Needle?
The answer is no.
Briefly explain the reasoning behind this answer.
The Federal Reserve has multiple regional branches, but the closest one relevant to Seattle would be the Federal Reserve Bank of San Francisco, which has a Seattle branch. However, it is located in downtown Seattle several blocks from the Space Needle—about a 15–20 minute walk—so it's not a "quick walk." """,

    """Can any person with a driver's license work in transport of aviation fuel?
The answer is no.
Briefly explain the reasoning behind this answer.
Not everyone with a driver's license can transport aviation fuel because it is classified as a hazardous material. Transporting it requires specialized training, certification (such as a HAZMAT endorsement in many countries), and adherence to strict safety and regulatory standards beyond ordinary driving privileges."""
]
    sampled_items = random.sample(examples_list, n)
    return "\n\n".join(str(item) for item in sampled_items) + "\n"






class CritiqueStringContainer:
    def __init__(self, dataset_name: str = 'cqa'):
        self.dataset_name = dataset_name
        if dataset_name == "cqa":
            self.POSITIVE_RATIONALES = self.POSITIVE_RATIONALES_CQA
            self.CRITIQUE_EXAMPLES = self.CRITIQUE_EXAMPLES_CQA
            self.REVISION_EXAMPLES = self.REVISION_EXAMPLES_CQA
        elif dataset_name == "esnli":
            self.POSITIVE_RATIONALES = self.POSITIVE_RATIONALES_ESNLI
            self.CRITIQUE_EXAMPLES = self.CRITIQUE_EXAMPLES_ESNLI
            self.REVISION_EXAMPLES = self.REVISION_EXAMPLES_ESNLI
        elif dataset_name == "strategyqa":
            self.POSITIVE_RATIONALES = self.POSITIVE_RATIONALES_STRATEGYQA
            self.CRITIQUE_EXAMPLES = self.CRITIQUE_EXAMPLES_STRATEGYQA
            self.REVISION_EXAMPLES = self.REVISION_EXAMPLES_STRATEGYQA
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Build the few-shot examples
        self._build_few_shot_examples()
        
    POSITIVE_RATIONALES_CQA = (
        'The answer is blotter. A blotter is a specially designed absorbent paper or material used to absorb excess ink when writing with a fountain pen. It prevents the ink from smudging on the paper and helps keep the writing clean and neat. Using a blotter is a common practice when using fountain pens to maintain the quality of the written work and prevent ink from transferring onto other surfaces or items.',
        'The answer is television because televisions often require cable connections for receiving cable TV signals, enabling access to a wide range of channels and entertainment content. This connection is essential for viewers who want access to cable programming on their TV screens.',
        'The answer is natural habitat because foxes typically seek refuge in forests, which provide them with suitable shelter, food, and cover from predators. This choice aligns with their natural behavior and survival instincts.',
        'The answer is populated areas because Sammy\'s desire to go where the people were suggests he should head to locations with a high population density, such as cities, towns, or crowded public places, which aligns with the definition of populated areas as places with a lot of people.',
        'The answer is grocery cart because it is the appropriate location within a store where customers temporarily place their items before proceeding to the checkout area. This ensures efficient and organized shopping and checkout processes.',
        'The answer is atlas. Google Maps and other GPS services have replaced traditional paper atlases as they provide more accurate and up-to-date geographical information, making them a more convenient and reliable option for navigation.',
        'The answer is bitterness. This choice aligns with the wife\'s resentful feeling of anger, which arises from being unfairly burdened with all the work in the marriage, thus indicating a sense of inequity and frustraction in the relationship.'
    )
    
    POSITIVE_RATIONALES_ESNLI = (
        'The answer is neutral. The premise confirms that the man is playing with the baby and making him laugh, but it does not specify how he is playing. The hypothesis adds the specific detail that he is playing peekaboo, which is not entailed by the premise and could be false without contradicting it.',
        'The answer is entailment. The premise explicitly states that a woman balances a pie on her head, which necessarily means she has "something" (the pie) on her head. The hypothesis is a generalization of this fact, making the relationship entailment.',
        'The answer is contradiction. The premise describes only people outside the building, with no mention of anyone inside or at the windows. The hypothesis asserts the presence of people inside looking out from windows, which contradicts the premise.',
        'The answer is entailment. The premise describes people "trying to climb a cliff," which necessarily implies they are physically on the cliff face or in contact with it. The hypothesis states simply that "people are on the cliff," which is fully contained within the premise.',
        'The answer is neutral. The premise only states that the men are talking, without specifying the topic. The hypothesis introduces a specific subject that is not entailed by the premise - it\'s possible but not necessary.',
        'The answer is contradiction. The premise states a dog is running through a field, while the hypothesis claims a cat is sleeping. These describe entirely different animals performing different activities, making them mutually contradictory.',
        'The answer is entailment. The premise describes children playing on swings in a playground, which necessarily means there are children present in that location. The hypothesis stating "children are in the playground" is logically guaranteed by the premise.'
    )
    
    POSITIVE_RATIONALES_STRATEGYQA = (
        'The answer is no. Kangaroos are marsupials, not birds or reptiles. They give birth to live young that develop in a pouch rather than laying eggs that need incubation.',
        'The answer is no. The Atlantic hurricane season runs from June 1 to November 30, with peak activity between mid-August and late October. The Gulf of Mexico remains susceptible to tropical cyclones during Halloween time.',
        'The answer is no. Grand Theft Auto III contains extensive depictions of illegal activities including attacks on police, which conflict with the pro-law enforcement values central to Blue Lives Matter.',
        'The answer is no. Rede Globo is a Brazilian network that primarily broadcasts in Portuguese, Brazil\'s official language. There\'s no institutional reason for anchors to speak Chinese on-air.',
        'The answer is yes. This is based on wordplay: Darwin developed the theory of evolution, and in Pokémon, Pikachu evolves into Raichu. The connection is that Darwin\'s work underpins the concept of evolution that Pikachu experiences.',
        'The answer is no. The Federal Reserve has multiple regional branches, but the closest to Seattle would be the San Francisco Fed\'s Seattle branch, located downtown several blocks from the Space Needle—about a 15–20 minute walk, not a "quick walk."',
        'The answer is no. Not everyone with a driver\'s license can transport aviation fuel because it\'s classified as hazardous material, requiring specialized training, certification (like HAZMAT endorsement), and strict regulatory compliance beyond ordinary driving privileges.'
    )


    CRITIQUE_EXAMPLES_CQA = (
                "The rationale is generally correct, helpful, clear, concise, comprehensive, informative, and understandable. However, there is a minor improvement that can be made to differentiate it further. Specifically, the rationale could be improved by providing a brief explanation of why the other options (a) shirt pocket, (b) calligrapher’s hand, (c) inkwell, and (d) desk drawer are incorrect. This would help the user understand not only why 'blotter' is the correct answer but also why the other options are not suitable for absorbing extra ink from a fountain pen.",
                "The rationale is not correct because it makes an assumption that all televisions require cable connections for home entertainment, which is not accurate. Some televisions can access entertainment content through methods other than cable, such as streaming services, satellite, or over-the-air broadcasts. The rationale is not comprehensive because it does not acknowledge other methods of accessing entertainment content on a television besides cable. Also, it is not differentiating because it doesn't distinguish between different types of home entertainment equipment. While it correctly identifies television as the answer, it does not explain why the other options (radio shack, substation, cabinet, desk) are incorrect, which would have been informative to the user.",
                "The rationale doesn't clearly explain why the fox might be looking for its 'natural habitat'. It assumes that the fox is searching for its habitat without specifying why it might have left the city in the first place. It assumes that the fox is looking for a habitat, but the question itself does not necessarily imply this. The question is open-ended and does not provide information about the fox's intentions. While it briefly mentions that forests provide shelter and food for foxes, it doesn't delve into the other potential reasons why the fox might have walked into the forest, such as searching for prey or a mate. The rationale doesn't differentiate the chosen answer ('natural habitat') from the other options, such as why it is a better choice than the alternatives like 'hen house' or 'dense forest'. This lack of differentiation is crucial for clarity.",
                "While the rationale correctly identifies the answer as 'populated areas', it doesn't explore or mention any potential counterarguments or alternative interpretations. It could be improved by briefly addressing why the other options (b) race track, (c) desert, (d) apartment, and (e) roadblock are not suitable for Sammy's goal. This would make the explanation more comprehensive and helpful. The rationale doesn't effectively differentiate the chosen answer from the other options. It could be improved by explicitly highlighting what makes 'populated areas' the best choice compared to the other options. For example, mentioning that populated areas have a high concentration of people, which aligns with Sammy's desire, would provide a stronger differentiation.",
                "The rationale could be clearer in explaining why the grapes should go in the grocery cart. It doesn't explicitly state that the grapes should be placed in the cart to be purchased. This might confuse someone unfamiliar with shopping procedures. The explanation could be more concise. It includes unnecessary information about efficient and organized shopping and checkout processes, which is not directly related to the question. The rationale doesn't differentiate the choice 'grocery cart' from the other options. It could highlight why the grocery cart is the most appropriate choice compared to the alternatives (e.g., why it's better than putting grapes in one's mouth or in a fruit basket)",
                "The rationale provided by the assistant is clear and concise. However, it lacks comprehensiveness and differentiation: While the rationale correctly identifies that Google Maps and GPS services have replaced atlases, it doesn't provide any contrasting information or alternatives that could make the response more informative. For example, it could mention briefly how digital mapping services offer real-time updates and interactive features that paper atlases cannot provide. Lack of Comprehensiveness: The rationale doesn't delve into the full range of reasons why Google Maps and GPS services have replaced atlases. It could include additional points such as the convenience of GPS services, the ability to search for specific locations, and the integration with other useful features like traffic information. Overall, while the response is correct and clear, it could be enhanced by providing a more comprehensive and differentiating explanation of why atlases have been replaced by digital mapping services like Google Maps.",
                "The rationale is generally correct, but there are some areas where it could be improved: Conciseness: The explanation is somewhat verbose. It could be more concise to enhance clarity and readability. Comprehensiveness: While the explanation does touch on the wife's feelings of inequity and frustration, it could provide a bit more detail or context to fully explain why 'bitterness' is the most appropriate choice. Differentiation: The explanation could highlight why 'bitterness' is a more suitable choice compared to the other options. What distinguishes it from the other emotions listed? Clarity: While the explanation is generally clear, it could benefit from a clearer connection between 'bitterness' and the wife's feelings. Why does being unfairly burdened lead to bitterness specifically? Helpfulness: Adding a real-life example or a brief hypothetical scenario could make the explanation more relatable and helpful to the reader in understanding the choice. Informativeness: The explanation could briefly mention that bitterness is often associated with long-standing resentment and a sense of injustice, which aligns with the situation described in the question. Overall, the rationale is on the right track but could be improved in terms of conciseness, clarity, and the depth of explanation to make it more informative and helpful."
    )
    
    CRITIQUE_EXAMPLES_ESNLI = (
        "The answer is mostly correct but unhelpful because it doesn’t clearly differentiate *why* the hypothesis cannot be entailed. It vaguely says “possible but not necessary” without explaining that entailment requires the hypothesis to be guaranteed by the premise. It also misses an opportunity to clarify why the relationship is not contradiction (the premise doesn’t rule out erectile dysfunction as a topic). Overall, it’s accurate but somewhat vague, repetitive, and not sharply informative.",
        "The answer is mostly correct, but it could be more concise and focused. It spends unnecessary space restating the premise and hypothesis rather than directly justifying the entailment. The explanation is somewhat verbose for a simple logical relationship, making it feel unhelpfully padded. It also doesn’t clearly distinguish why the relationship is entailment instead of neutral, which might leave ambiguity for someone new to textual entailment.",
        "The answer is unconvincing because it misclassifies the relation: the premise does not rule out the possibility of people inside looking out, it merely makes no claim about them. Thus, “contradiction” is inaccurate; the correct label is “neutral.” The reasoning is also misleading, since it treats omission as contradiction rather than absence of entailment. It lacks nuance in distinguishing between contradiction and neutrality.",
        "The answer is unconvincing because it assumes that “trying to climb a cliff” necessarily means being physically on the cliff, whereas the phrase could also describe people preparing to climb or attempting from the base, not yet on it. The reasoning is overly narrow, failing to acknowledge this ambiguity. It is also vague in claiming the hypothesis is “fully contained” in the premise without carefully distinguishing between being “on” a cliff versus “near” or “attempting to get onto” one.",
        "The answer is mostly correct but somewhat unhelpful because it repeats the premise and hypothesis without clearly distinguishing why 'peekaboo' introduces new information. It is a bit wordy and vague—saying the hypothesis 'could be false without contradicting it' is accurate but not very precise in explaining why this makes the relation *neutral*. It also misses the chance to explicitly contrast entailment and neutrality, which would make the reasoning more convincing and differentiated.",
        "The answer is inaccurate and unconvincing because the premise and hypothesis describe unrelated situations, not mutually exclusive ones. A dog running does not prevent a cat from sleeping, so the correct relationship is *neutral*, not contradiction. The explanation is also misleading and vague, as it wrongly assumes different animals doing different actions automatically creates contradiction, without clarifying why contradiction requires logical impossibility rather than simple difference.",
        "The answer is generally correct but somewhat unhelpful because it repeats the obvious without adding much reasoning depth. It is vague in not distinguishing *why* the premise fully guarantees the hypothesis beyond simply restating it. It is also slightly non-differentiating, since it doesn’t contrast entailment with the other options (neutral, contradiction), which would make the justification stronger. Overall, while accurate, the explanation feels a bit shallow and uninformative."
    )
    
    CRITIQUE_EXAMPLES_STRATEGYQA = (
        "The answer is factually correct but unhelpful in several ways. It defines “incubate” only in terms of eggs, ignoring that the question might be probing whether the pouch functions analogously to incubation. The phrasing “the answer is no” is abrupt and uninformative, offering no nuance about marsupial reproduction. The explanation about birds and reptiles is tangential and non-differentiating, since the real contrast should be between egg incubation and pouch-based postnatal development. Finally, the response misses an opportunity to clarify that the pouch provides warmth, protection, and nutrition, which are incubation-like functions, though not technically incubation.",
        "The answer is unhelpful because it gives an absolute “no” without nuance, when risk is probabilistic rather than categorical. It is ambiguous in failing to distinguish between “likely” and “possible,” conflating seasonal susceptibility with actual storm probability for a given year. It is misleading in implying that all cruises around Halloween are unsafe, rather than explaining relative risk levels or historical storm frequency. It is vague and uninformative because it does not provide context such as the likelihood of storms in late October versus earlier in the season, nor does it mention that cruise operators monitor storms closely and adjust itineraries. Overall, the answer is unconvincing because it lacks data, nuance, and differentiation between hazard and risk management.",
        "The answer is overly simplistic and vague: it asserts “the answer is no” without analyzing nuances such as whether some themes (e.g., law enforcement presence or crime prevention narratives) might align with Blue Lives Matter views. It is unconvincing because it provides only one reason—attacks on police—while GTA III includes many other elements (organized crime, civilian violence, corruption) that could be objectionable. It is also non-differentiating, since the reasoning could apply to almost any pro-police group, not specifically Blue Lives Matter, leaving the response generic and underdeveloped.",
        "The answer is too categorical and dismissive: saying “the answer is no” ignores the possibility that some anchors might personally know Chinese, even if they don’t use it professionally. It conflates the institutional broadcasting language with individual language abilities, which makes the reasoning somewhat illogical. It is also uninformative in that it does not clarify whether Rede Globo has ever aired Chinese-language content, for example in special reports or interviews. Finally, it lacks nuance and fails to distinguish between “speaking Chinese at all” and “speaking Chinese on-air.”",
        "The answer is unhelpful because it treats a fictional character’s “liking” of a historical figure as literal, which is illogical and misleading. It conflates scientific evolution with the game mechanic of Pokémon “evolution,” which is conceptually unrelated, making the reasoning inaccurate and unconvincing. The wordplay is presented as if it were factual, creating ambiguity about whether this is meant as a joke or a serious claim. Overall, it lacks clarity, precision, and differentiation between metaphorical playfulness and actual knowledge.",
        "The answer is partly misleading and imprecise. It conflates the Federal Reserve’s regional structure with the user’s practical question about walking distance, introducing unnecessary institutional detail instead of directly addressing location. Calling it “not a quick walk” but then saying it’s a 15–20 minute walk is ambiguous and subjective, since some would consider that a quick walk. The phrasing “closest to Seattle would be the San Francisco Fed’s Seattle branch” is confusing, as it suggests the San Francisco Fed is in Seattle rather than clarifying it operates a local office. Overall, the explanation obscures a straightforward fact about walking distance with unnecessary complexity and imprecise language.",
        "The answer is accurate in its conclusion but unhelpful in several ways: it is vague about which authorities regulate aviation fuel transport (e.g., FAA, DOT, ICAO), it does not specify the exact requirements such as a CDL with HAZMAT endorsement in the U.S., it conflates 'any person with a driver’s license' with those who might already hold commercial licenses, and it lacks differentiation between private handling and commercial transport contexts. This makes the response clear at a high level but incomplete, ambiguous, and unconvincing for an expert reader seeking precise regulatory detail."
    )

    INSTRUCTION_CRITIQUE = "CritiqueRequest: Identify specific ways in which the answer is unhelpful, illogical, ambiguous, misleading, inaccurate, vague, non-differentiating, unconvincing, unclear, or uninformative."

    INSTRUCTION_REVISION = "RevisionRequest: Please rewrite the answer to eliminate any and all incorrect, unhelpful, unclear, unconcise, non-comprehensive, uninformative, not understandable, or non-differentiating elements."

    CRITIQUE_REVISION_REQUEST_INSTRUCTION_PAIRS = (
                ("CritiqueRequest: Identify specific ways in which the answer is incorrect, unhelpful, unclear, unconcise, non-comprehensive, uninformative, not understandable, or non-differentiating.",
                "RevisionRequest: Please rewrite the answer to eliminate any and all incorrect, unhelpful, unclear, unconcise, non-comprehensive, uninformative, not understandable, or non-differentiating elements."),
                ("CritiqueRequest: Explain ways in which the answer may be incorrect or unhelpful. For example, if the assistant includes anything that encourages unconcise or non-comprehensive reasoning, please highlight it. Other examples of issues include lack of clarity or informativeness.",
                "RevisionRequest: Please revise the answer to eliminate all incorrect and unhelpful elements, ensuring it is clear, concise, comprehensive, informative, and understandable, while avoiding unconcise or non-differentiating content."),
                ("CritiqueRequest: Identify all ways in which the answer is incorrect, unhelpful, or lacks differentiation. Furthermore, provide specific details on how the assistant can improve its answer.",
                "RevisionRequest: Please rewrite the answer to remove all incorrect, unhelpful, or non-differentiating content and guide the answer in a more reliable and understandable direction."),
                ("CritiqueRequest: Identify specific ways in which the preceding answer is incorrect. Examples of problems include unconcise reasoning or a lack of differentiation.",
                "RevisionRequest: Rewrite the answer to remove all incorrect content and address any problematic assumptions, such as unconcise or non-differentiating reasoning."),
                ("CritiqueRequest: Discuss the answer and whether it is in any way unclear, unconcise, or non-differentiating.",
                "RevisionRequest: Revise the answer to eliminate content that is unclear, unconcise, or non-differentiating."),
                ("CritiqueRequest: Discuss whether the answer included anything that may be interpreted as unconcise, unhelpful, or lacking differentiation.",
                "RevisionRequest: Please write a new version of the answer that removes unconcise and unhelpful content and addresses any problematic assumptions from the human."),
                ("CritiqueRequest: What's the strongest argument you could give that the answer was in some way incorrect, unhelpful, unclear, unconcise, non-comprehensive, uninformative, not understandable, or non-differentiating?",
                "RevisionRequest: Amend the answer to remove any and all incorrect, unhelpful, unclear, unconcise, non-comprehensive, uninformative, not understandable, or non-differentiating content."),
                ("CritiqueRequest: Take a look at the answer and note if it was in any way non-comprehensive or lacked differentiation based on common-sense reasoning and logical thinking.",
                "RevisionRequest: Please write a new response that is comprehensive and does not lack differentiation based on common-sense reasoning and logical thinking."),
                ("CritiqueRequest: Read the answer carefully and identify anything it said that provides reasoning or guidance that may be interpreted as non-comprehensive or lacking differentiation.",
                "RevisionRequest: Rewrite the answer to provide comprehensive and helpful reasoning, avoiding any shortcomings in this regard."),
                ("CritiqueRequest: Critique the answer and identify anything it said that was incorrect, unhelpful, or lacked differentiation.",
                "RevisionRequest: Please rewrite the answer to remove any and all incorrect, unhelpful, or non-differentiating content.")
        )


    REVISION_EXAMPLES_CQA = (
                """A blotter is a specially designed absorbent paper or material used to absorb excess ink when writing with a fountain pen. It prevents the ink from smudging on the paper and helps keep the writing clean and neat. To clarify why the other options are incorrect: (a) shirt pocket: Using a shirt pocket to absorb ink would likely result in a mess and could stain the clothing, (b) calligrapher’s hand: The hand of the calligrapher would not effectively absorb ink, and it would still lead to smudging, (c) inkwell: An inkwell is a container for holding ink, not for absorbing it, (d) desk drawer: A desk drawer is not designed to absorb ink; it's meant for storage and organization. So the answer is blotter.""",
                """Televisions often require cable connections for receiving cable TV signals, enabling access to a wide range of channels and entertainment content. However, it's important to note that not all televisions require cable connections for home entertainment. As for the other options: (a) Radio shack - This is not home entertainment equipment; it's a retail store for electronics. (b) Substation - This is not home entertainment equipment; it typically refers to a facility for electrical power distribution. (c) Cabinet - This is a piece of furniture and not home entertainment equipment. (e) Desk - This is also a piece of furniture and not home entertainment equipment. So, the correct answer is (d) television.""",
                """The fox's intention for walking from the city into the forest is not explicitly mentioned in the question. However, we can make an educated guess based on typical fox behavior. Foxes are known to seek out natural habitats like forests for several reasons. While we cannot definitively state its purpose, here are some plausible explanations: (a) Pretty flowers: Foxes are not typically known for seeking flowers, so this option is less likely. (b) Hen house: Foxes might be attracted to hen houses for easy prey, but this option assumes the fox is actively looking for one. (c) Natural habitat: This choice aligns with the natural behavior of foxes, as forests provide shelter, food, and cover from predators. (d) Storybook: This option is likely unrelated to the fox's actions. (e) Dense forest: This choice is similar to (c) but lacks the specificity of 'natural habitat'. Given the limited information provided, (c) "natural habitat" is the most plausible answer, as it aligns with typical fox behavior.""",
                """Sammy's desire to be where the people are indicates a preference for locations with a high population density. Populated areas, such as cities, towns, or crowded public places, best align with this preference. Let's briefly consider the other options: (b) Race track: Race tracks are generally places for specific events, not necessarily where people gather in large numbers regularly. (c) Desert: Deserts are sparsely populated, making them an unlikely choice for someone seeking company. (d) Apartment: While apartments can have people, they are individual living spaces and do not guarantee the presence of many people in one place. (e) Roadblock: Roadblocks are temporary obstructions and do not represent places where people gather. In summary, "populated areas" stands out as the most suitable choice because it directly addresses Sammy's desire to be where people congregate, distinguishing it from the other options.""",
                """The grapes should be placed in the grocery cart. This is the right choice because the grocery cart is where customers typically keep their items while shopping before they eventually check out. Placing the grapes in the grocery cart ensures that they are ready for purchase and helps keep your shopping organized. This option is more appropriate than putting grapes in your mouth or a fruit basket, as it aligns with the usual shopping process and facilitates a smooth checkout experience.""",
                """Google Maps and other GPS services have replaced traditional paper atlases for several reasons. Firstly, digital mapping services offer real-time updates, ensuring that users have access to the most current and accurate geographical information. Secondly, Google Maps and GPS services provide a level of interactivity and convenience that paper atlases cannot match e.g., to search for. This makes navigation far more efficient and user-friendly. Furthermore, digital mapping services often integrate additional features such as live traffic information, alternate route suggestions, and street-level imagery. These features enhance the overall navigation experience, making it easier for users to plan their journeys effectively. In summary, the transition from atlases to digital mapping services like Google Maps has been driven by the real-time updates, interactive capabilities, and the integration of helpful features that offer users a more comprehensive and efficient way to navigate their surroundings. Hence the answer is (d) atlas. """,
                """Bitterness is the most suitable choice because it reflects the wife's deep-seated resentment and frustration stemming from the unfair burden of doing all the work in the marriage. This emotion often arises from a prolonged sense of injustice and inequality, making it a distinct and apt choice in this context. Bitterness is characterized by a lingering sense of grievance and an underlying anger that arises when someone feels unfairly treated or taken advantage of. In this scenario, the wife likely felt increasingly embittered as she shouldered most of the responsibilities in the relationship without receiving the support or effort from her partner that she expected. This emotional state can be a significant factor leading to divorce, as it indicates a breakdown in the marital relationship due to unaddressed grievances and a lack of equity."""
    )
    
    REVISION_EXAMPLES_ESNLI = (
        "The premise states that a man in a striped shirt is playing with a baby and making him laugh. The hypothesis specifies that the man is playing peekaboo. While the hypothesis is consistent with the premise, the premise does not explicitly confirm that the game is peekaboo. Because the hypothesis introduces a specific detail not mentioned in the premise, it is neither necessarily true nor necessarily false based on the premise alone. This makes the relationship neutral, not entailment or contradiction.",
        "The premise states that a woman balances a pie on her head. This directly implies that she has an object (the pie) on her head. The hypothesis—“The woman has something on her head”—is a general statement that is guaranteed to be true if the premise is true. Therefore, the truth of the premise logically ensures the truth of the hypothesis, which is the defining criterion for entailment. It is not neutral, because the premise does guarantee the hypothesis, and it is not a contradiction, because there is no conflict between the statements.",
        "The premise states only that people are walking around outside of a building. It gives no information about whether anyone is inside the building or looking out of its windows. The hypothesis introduces a new detail—people looking outside from the windows—that is neither confirmed nor denied by the premise. Because the premise does not address this possibility, the relationship is neutral.",
        "The premise states that people are 'trying to climb a cliff,' which indicates an attempt or effort to climb. However, it does not guarantee that they have successfully reached or are physically on the cliff. The hypothesis, 'People are on the cliff,' asserts a fact that is not fully confirmed by the premise. Therefore, the correct relationship is neutral—the hypothesis may be true, but it is not necessarily entailed by the premise.",
        "The premise states that a group of men are sitting at a table and talking but does not specify the topic of conversation. The hypothesis claims they are talking about erectile dysfunction, which is a specific subject not guaranteed by the premise. In Natural Language Inference (NLI) terms, this is neutral because the hypothesis might be true or might be false given the premise. Entailment would require the premise to guarantee the truth of the hypothesis. Contradiction would require the premise to make the hypothesis impossible. Neutral means the premise provides no definitive information about the truth of the hypothesis, which is the case here.",
        "In natural language inference, contradiction occurs when the premise and hypothesis describe mutually incompatible situations that cannot both be true simultaneously. The premise describes 'a dog running through a field' - specifying the animal (dog), action (running), and location (field). The hypothesis states 'a cat is sleeping' - different animal (cat), different action (sleeping), and potentially different context. These scenarios describe entirely different situations with incompatible elements, making it impossible for both to be describing the same scene. Therefore, the relationship is contradiction.",
        "The premise states that 'children are playing on swings in a playground.' This explicitly situates the children within the playground while they engage in an activity. The hypothesis, 'children are in the playground,' is fully guaranteed by the premise because the premise already confirms both the presence of children and their location. It cannot be neutral or contradictory, since the premise directly supports the hypothesis rather than leaving it uncertain or opposing it. Therefore, the correct relationship is entailment."
    )
    
    REVISION_EXAMPLES_STRATEGYQA = (
        "Kangaroos do not incubate their offspring in the way birds or reptiles do with eggs. Instead, as marsupials, they give birth to extremely underdeveloped live young that continue developing inside the mother’s pouch. The pouch provides warmth, protection, and access to milk, which supports the young in a way that is functionally similar to incubation, even though it is postnatal rather than egg-based.",
        "A Halloween cruise in the Gulf of Mexico carries some risk from tropical storms, but it is not automatically unsafe. The Atlantic hurricane season runs from June 1 to November 30, with peak activity typically from mid-August through late October. By late October, the probability of storms is declining compared to the peak, though the Gulf can still experience tropical cyclones. Cruise lines actively monitor weather forecasts and adjust itineraries or ports of call to avoid severe weather, which significantly reduces actual risk for passengers. Historically, while storms in late October do occur, the majority of cruises complete without storm-related disruption. Travelers should stay informed, have travel insurance, and be prepared for possible itinerary changes. In short, while some risk exists, Halloween cruises in the Gulf are generally considered manageable with proper precautions and monitoring.",
        "Members of Blue Lives Matter would likely find significant aspects of Grand Theft Auto III objectionable, though not necessarily every element. The game’s open-world gameplay revolves around criminal activities, including theft, organized crime, and frequent acts of violence, some of which directly target police officers. These elements conflict with the group’s core pro-law enforcement and public safety values. However, the game also depicts law enforcement presence, criminal investigations, and the consequences of crime, which might resonate with a subset of the group’s perspectives on law and order. Overall, while no member would probably endorse the game’s criminal behaviors, individual reactions could vary depending on how they weigh depictions of police authority versus depictions of crime.",
        "Rede Globo is a Brazilian television network that primarily broadcasts in Portuguese, which is Brazil’s official language. Its news anchors generally present content in Portuguese, and there is no standard practice of speaking Chinese on-air. While it is theoretically possible that some individual anchors might know Chinese personally, this knowledge is not part of their professional duties. Occasionally, Rede Globo may feature Chinese-language content in special reports, interviews, or international coverage, but this would typically be translated or interpreted for the Portuguese-speaking audience.",
        "Pikachu, as a fictional Pokémon character, does not have the capacity to like or dislike historical figures such as Charles Darwin. Any connection between Pikachu and Darwin is purely a humorous wordplay: Darwin is famous for the theory of evolution, and in Pokémon, Pikachu “evolves” into Raichu. This similarity is metaphorical and not indicative of Pikachu having any actual opinion about Darwin.",
        "No, there is no Federal Reserve bank located within a short walking distance of the Space Needle. The nearest Federal Reserve presence in Seattle is a branch office downtown, roughly a 15–20 minute walk from the Space Needle.",
        "No, not every person with a driver’s license is legally permitted to transport aviation fuel. Aviation fuel is classified as a hazardous material, and its transport is strictly regulated. In the United States, for example, commercial transport of aviation fuel requires a Commercial Driver’s License (CDL) with a HAZMAT endorsement, as regulated by the Department of Transportation (DOT) and enforced by the Federal Motor Carrier Safety Administration (FMCSA). Drivers must complete specialized hazardous materials training and pass background checks. Similar regulations apply internationally, guided by organizations such as the International Civil Aviation Organization (ICAO) and national aviation authorities. Transporting aviation fuel without the proper certification is illegal and poses significant safety and legal risks."
    )
        

    def _build_few_shot_examples(self):
        """Build dataset-specific few-shot examples"""
        if self.dataset_name == "cqa":
            self.FEW_SHOT_EXAMPLES_CRITIQUE_REQUEST = (
                f"Question: What do people use to absorb extra ink from a fountain pen?\nChoices: (a) shirt pocket (b) calligrapher's hand (c) inkwell (d) desk drawer (e) blotter \nAnswer: {self.POSITIVE_RATIONALES[0]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[0]}",
                f"Question: What home entertainment equipment requires cable?\nChoices: (a) radio shack (b) substation (c) cabinet (d) television (e) desk \nAnswer: {self.POSITIVE_RATIONALES[1]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[1]}",
                f"Question: The fox walked from the city into the forest, what was it looking for?\nChoices: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook (e) dense forest \nAnswer: {self.POSITIVE_RATIONALES[2]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[2]}",
                f"Question: Sammy wanted to go to where the people were. Where might he go?\nChoices: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock \nAnswer: {self.POSITIVE_RATIONALES[3]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[3]}",
                f"Question: Where do you put your grapes just before checking out?\nChoices: (a) mouth (b) grocery cart (c) super market (d) fruit basket (e) fruit market \nAnswer: {self.POSITIVE_RATIONALES[4]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[4]}",
                f"Question: Google Maps and other highway and street GPS services have replaced what?\nChoices: (a) united states (b) mexico (c) countryside (d) atlas (e) oceans \nAnswer: {self.POSITIVE_RATIONALES[5]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[5]}",
                f"Question: Before getting a divorce, what did the wife feel who was doing all the work?\nChoices: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness \nAnswer: {self.POSITIVE_RATIONALES[6]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[6]}"
            )
            self.FEW_SHOT_EXAMPLES_REVISION_REQUEST = (
                f"Question: What do people use to absorb extra ink from a fountain pen?\nChoices: (a) shirt pocket (b) calligrapher's hand (c) inkwell (d) desk drawer (e) blotter \nAnswer: {self.POSITIVE_RATIONALES[0]}\nCritique: {self.CRITIQUE_EXAMPLES[0]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[0]}",
                f"Question: What home entertainment equipment requires cable?\nChoices: (a) radio shack (b) substation (c) cabinet (d) television (e) desk \nAnswer: {self.POSITIVE_RATIONALES[1]}\nCritique: {self.CRITIQUE_EXAMPLES[1]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[1]}",
                f"Question: The fox walked from the city into the forest, what was it looking for?\nChoices: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook (e) dense forest \nAnswer: {self.POSITIVE_RATIONALES[2]}\nCritique: {self.CRITIQUE_EXAMPLES[2]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[2]}",
                f"Question: Sammy wanted to go to where the people were. Where might he go?\nChoices: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock \nAnswer: {self.POSITIVE_RATIONALES[3]}\nCritique: {self.CRITIQUE_EXAMPLES[3]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[3]}",
                f"Question: Where do you put your grapes just before checking out?\nChoices: (a) mouth (b) grocery cart (c) super market (d) fruit basket (e) fruit market \nAnswer: {self.POSITIVE_RATIONALES[4]}\nCritique: {self.CRITIQUE_EXAMPLES[4]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[4]}",
                f"Question: Google Maps and other highway and street GPS services have replaced what?\nChoices: (a) united states (b) mexico (c) countryside (d) atlas (e) oceans \nAnswer: {self.POSITIVE_RATIONALES[5]}\nCritique: {self.CRITIQUE_EXAMPLES[5]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[5]}",
                f"Question: Before getting a divorce, what did the wife feel who was doing all the work?\nChoices: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness \nAnswer: {self.POSITIVE_RATIONALES[6]}\nCritique: {self.CRITIQUE_EXAMPLES[6]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[6]}"
            )
        elif self.dataset_name == "esnli":
            self.FEW_SHOT_EXAMPLES_CRITIQUE_REQUEST = (
                f"Question: Premise: A man in a striped shirt is playing with a baby and making him laugh. Hypothesis: A man is playing pickaboo with a baby and making him laugh. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[0]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[0]}",
                f"Question: Premise: Woman balances a pie on her head and looks up into camera Hypothesis: The woman has something on her head What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[1]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[1]}",
                f"Question: Premise: People are walking around outside of a building. Hypothesis: There are people looking outside from the windows of the building. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[2]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[2]}",
                f"Question: Premise: People at a distance trying to climb a cliff. Hypothesis: People are on the cliff. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[3]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[3]}",
                f"Question: Premise: A group of men are sitting at a table talking. Hypothesis: A group of men are talking about erectile dysfunction. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[4]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[4]}",
                f"Question: Premise: A dog is running through a field. Hypothesis: A cat is sleeping. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[5]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[5]}",
                f"Question: Premise: Children are playing on swings in a playground. Hypothesis: Children are in the playground. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[6]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[6]}"
            )
            self.FEW_SHOT_EXAMPLES_REVISION_REQUEST = (
                f"Question: Premise: A man in a striped shirt is playing with a baby and making him laugh. Hypothesis: A man is playing pickaboo with a baby and making him laugh. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[0]}\nCritique: {self.CRITIQUE_EXAMPLES[0]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[0]}",
                f"Question: Premise: Woman balances a pie on her head and looks up into camera Hypothesis: The woman has something on her head What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[1]}\nCritique: {self.CRITIQUE_EXAMPLES[1]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[1]}",
                f"Question: Premise: People are walking around outside of a building. Hypothesis: There are people looking outside from the windows of the building. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[2]}\nCritique: {self.CRITIQUE_EXAMPLES[2]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[2]}",
                f"Question: Premise: People at a distance trying to climb a cliff. Hypothesis: People are on the cliff. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[3]}\nCritique: {self.CRITIQUE_EXAMPLES[3]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[3]}",
                f"Question: Premise: A group of men are sitting at a table talking. Hypothesis: A group of men are talking about erectile dysfunction. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[4]}\nCritique: {self.CRITIQUE_EXAMPLES[4]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[4]}",
                f"Question: Premise: A dog is running through a field. Hypothesis: A cat is sleeping. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[5]}\nCritique: {self.CRITIQUE_EXAMPLES[5]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[5]}",
                f"Question: Premise: Children are playing on swings in a playground. Hypothesis: Children are in the playground. What is the relationship?\nChoices: (a) entailment (b) neutral (c) contradiction\nAnswer: {self.POSITIVE_RATIONALES[6]}\nCritique: {self.CRITIQUE_EXAMPLES[6]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[6]}"
            )
        elif self.dataset_name == "strategyqa":
            self.FEW_SHOT_EXAMPLES_CRITIQUE_REQUEST = (
                f"Question: Does a kangaroo incubate its offspring?\nAnswer: {self.POSITIVE_RATIONALES[0]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[0]}",
                f"Question: Is a Halloween cruise in the Gulf of Mexico likely to be safe from storms?\nAnswer: {self.POSITIVE_RATIONALES[1]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[1]}",
                f"Question: Would members of Blue Lives Matter support every element of Grand Theft Auto III?\nAnswer: {self.POSITIVE_RATIONALES[2]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[2]}",
                f"Question: Do the anchors on Rede Globo speak Chinese?\nAnswer: {self.POSITIVE_RATIONALES[3]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[3]}",
                f"Question: Does Pikachu like Charles Darwin?\nAnswer: {self.POSITIVE_RATIONALES[4]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[4]}",
                f"Question: Is the Federal Reserve a quick walk from Space Needle?\nAnswer: {self.POSITIVE_RATIONALES[5]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[5]}",
                f"Question: Can any person with a driver's license work in transport of aviation fuel?\nAnswer: {self.POSITIVE_RATIONALES[6]}\n{self.INSTRUCTION_CRITIQUE}\nCritique: {self.CRITIQUE_EXAMPLES[6]}"
            )
            self.FEW_SHOT_EXAMPLES_REVISION_REQUEST = (
                f"Question: Does a kangaroo incubate its offspring?\nAnswer: {self.POSITIVE_RATIONALES[0]}\nCritique: {self.CRITIQUE_EXAMPLES[0]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[0]}",
                f"Question: Is a Halloween cruise in the Gulf of Mexico likely to be safe from storms?\nAnswer: {self.POSITIVE_RATIONALES[1]}\nCritique: {self.CRITIQUE_EXAMPLES[1]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[1]}",
                f"Question: Would members of Blue Lives Matter support every element of Grand Theft Auto III?\nAnswer: {self.POSITIVE_RATIONALES[2]}\nCritique: {self.CRITIQUE_EXAMPLES[2]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[2]}",
                f"Question: Do the anchors on Rede Globo speak Chinese?\nAnswer: {self.POSITIVE_RATIONALES[3]}\nCritique: {self.CRITIQUE_EXAMPLES[3]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[3]}",
                f"Question: Does Pikachu like Charles Darwin?\nAnswer: {self.POSITIVE_RATIONALES[4]}\nCritique: {self.CRITIQUE_EXAMPLES[4]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[4]}",
                f"Question: Is the Federal Reserve a quick walk from Space Needle?\nAnswer: {self.POSITIVE_RATIONALES[5]}\nCritique: {self.CRITIQUE_EXAMPLES[5]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[5]}",
                f"Question: Can any person with a driver's license work in transport of aviation fuel?\nAnswer: {self.POSITIVE_RATIONALES[6]}\nCritique: {self.CRITIQUE_EXAMPLES[6]}\n{self.INSTRUCTION_REVISION}\nRevision: {self.REVISION_EXAMPLES[6]}"
            )

    def get_random_critique_revision_request_pair(self):
        return random.sample(self.CRITIQUE_REVISION_REQUEST_INSTRUCTION_PAIRS, 1)[0]

    def get_random_critique_examples(self, n_examples):
        return tuple(random.sample(self.FEW_SHOT_EXAMPLES_CRITIQUE_REQUEST, n_examples))

    def get_random_revision_examples(self, n_examples):
        return tuple(random.sample(self.FEW_SHOT_EXAMPLES_REVISION_REQUEST, n_examples))






