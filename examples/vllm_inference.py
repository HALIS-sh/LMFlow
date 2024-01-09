import json
import os
import torch
import time
import socket
from functools import cached_property, partial
from typing import Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING, Any
from lmflow.pipeline.utils.vllm_utils import Counter
from lmflow.pipeline.utils.vllm_config import (CacheConfig, ParallelConfig, SchedulerConfig)
from lmflow.pipeline.utils.vllm_sequence import (SamplerOutput, Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutputs,
                           SequenceOutputs, SequenceStatus)
from lmflow.pipeline.utils.vllm_ouputs import RequestOutput
from lmflow.pipeline.utils.vllm_scheduleroutputs import SchedulerOutputs, PreemptionMode, Scheduler
from lmflow.pipeline.utils.vllm_logger import init_logger
from lmflow.pipeline.utils.vllm_core.vllm_block_manager import BlockSpaceManager
from lmflow.pipeline.utils.vllm_core.vllm_policy import PolicyFactory
from lmflow.pipeline.utils.vllm_input_metadata import InputMetadata
from lmflow.pipeline.utils.vllm_worker import Worker
from lmflow.pipeline.utils.vllm_transformer_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
#from lmflow.utils.vllm import LLM, SamplingParams
from transformers import HfArgumentParser
from lmflow.args import ModelArguments, AutoArguments, InferencerArguments , SamplingType
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.utils.vllm_ray_utils import RayWorker, initialize_cluster, ray

def _run_workers(
    workers,
    method: str,
    *args,
    get_all_outputs: bool = False,
    **kwargs,
) -> Any:
    """Runs the given method on all workers."""
    all_outputs = []
    for worker in workers:
        if parallel_config.worker_use_ray:
            executor = partial(worker.execute_method.remote, method)
        else:
            executor = getattr(worker, method)

        output = executor(*args, **kwargs)
        all_outputs.append(output)

    if parallel_config.worker_use_ray:
        all_outputs = ray.get(all_outputs)

    if get_all_outputs:
        return all_outputs

    # Make sure all workers have the same results.
    output = all_outputs[0]
    for other_output in all_outputs[1:]:
        assert output == other_output
    return output

_SAMPLING_EPS = 1e-5
args_temperature = 0
args_top_p = 1.0
args_num_beams = 5
args_max_tokens = 50
args_repetition_penalty = 1.0
args_local_rank = int(os.getenv("LOCAL_RANK", "0"))
block_size: int = 16
swap_space: int = 4  # GiB
gpu_memory_utilization: float = 0.90
max_num_seqs: int = 256
max_num_batched_tokens: Optional[int] = None
pipeline_parallel_size: int = 1
tensor_parallel_size: int = 1
worker_use_ray: bool = False
scheduler_config : SchedulerConfig
log_stats: bool = False
# Sample prompts.
prompts = [
    "Hello, my name is",
    #"The president of the United States is",
    #"The capital of France is",
    #"The future of AI is",
]
# Sequence groups in the WAITING state.
vllm_waiting: List[SequenceGroup] = []
# Sequence groups in the RUNNING state.
vllm_running: List[SequenceGroup] = []
# Sequence groups in the SWAPPED state.
vllm_swapped: List[SequenceGroup] = []
use_vllm_flag = input("Please choose whether use vllm service(True/False):")
try:
    use_vllm_flag = bool(eval(use_vllm_flag))
except (NameError, SyntaxError):
    print("输入无效，请输入 True 或 False")
    # 如果用户输入无法解析为 bool 值，这里可以设置一个默认值，或者进行其他处理
    use_vllm_flag = False 
request_counter = Counter()
seq_counter = Counter()
logger = init_logger(__name__)
model_path = "output_models/gpt2"
pipeline_name = "inferencer"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
parser = HfArgumentParser((
    ModelArguments,
    PipelineArguments,
))
model_args, pipeline_args = (parser.parse_args_into_dataclasses())
inference_args = pipeline_args
with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)
model = AutoModel.get_model(
    model_args,
    tune_strategy='none',
    ds_config=ds_config,
    device=pipeline_args.device,
)
tokenizer = get_tokenizer(model_args.tokenizer_name, 
                        tokenizer_mode="auto",
                        trust_remote_code=model_args.trust_remote_code,
                        tokenizer_revision=model_args.tokenizer_revision,
                        revision=model_args.model_revision,)
cache_config = CacheConfig(block_size, gpu_memory_utilization, swap_space) # sliding_window:Optional
scheduler_config = SchedulerConfig(max_num_batched_tokens, max_num_seqs, model_args.max_model_len) # max_model_len:Optional
parallel_config = ParallelConfig(pipeline_parallel_size, tensor_parallel_size, worker_use_ray)
distributed_init_method, placement_group = initialize_cluster(parallel_config)
if use_vllm_flag:
    assert parallel_config.world_size == 1, ("Ray is required if parallel_config.world_size > 1.")
    workers: List[Worker] = []
    worker = Worker(model_args, parallel_config, scheduler_config, 0, distributed_init_method)
    workers.append(worker)
    _run_workers(workers, "init_model", get_all_outputs=True)
    # init_cache
    num_blocks = _run_workers(workers, "profile_num_available_blocks", 
                                get_all_outputs=True,
                                block_size=cache_config.block_size,
                                gpu_memory_utilization=cache_config.gpu_memory_utilization,
                                cpu_swap_space=cache_config.swap_space_bytes,)
    num_gpu_blocks = min(b[0] for b in num_blocks)
    num_cpu_blocks = min(b[1] for b in num_blocks)
    logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                f"# CPU blocks: {num_cpu_blocks}")
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                            "Try increasing `gpu_memory_utilization` when "
                            "initializing the engine.")

    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks
    # Initialize the cache.
    _run_workers(workers, "init_cache_engine", cache_config=cache_config)
    # init_cache end
    scheduler = Scheduler(scheduler_config, cache_config)
    prompt_limit = min(scheduler_config.max_model_len, scheduler_config.max_num_batched_tokens)
    block_manager = BlockSpaceManager(cache_config.block_size, 
                                    num_gpu_blocks=cache_config.num_gpu_blocks, 
                                    num_cpu_blocks=cache_config.num_cpu_blocks, 
                                    sliding_window=cache_config.sliding_window)
    policy = PolicyFactory.get_policy(policy_name="fcfs")



def main():
    # Create a sampling params object.
    #sampling_params = SamplingParams(temperature=0, top_p=1, use_beam_search=True, best_of=5, max_tokens=50, logprobs=1)
    #sampling_params = SamplingParams(temperature=0,  max_tokens=50, top_p=1)
    # Create an LLM.
    #llm = LLM(model="facebook/opt-125m")
    #llm = LLM(model=model_path)
    count = 1
    if use_vllm_flag:
        # process the input
        global prompts, request_counter, seq_counter
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        num_requests = len(prompts)
        #print("model_dtype:",model_args.vllm_dtype)
        #print(num_requests)
        for i in range(num_requests):
            prompt = prompts[i]
            request_id = str(next(request_counter))
            #prompt_token_ids = model.encode(prompt, return_tensors="pt").to(device=args_local_rank)
            prompt_token_ids = model.encode(prompt)
            # Create the sequences.
            block_size = cache_config.block_size
            seq_id = next(seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            #print("seq:", seq.data)
            # Create the sequence group
            arrival_time = time.monotonic()
            seq_group = SequenceGroup(request_id, [seq], inference_args, arrival_time)
            # add_request
            vllm_waiting.append(seq_group)
            # run_inference
            outputs: List[RequestOutput] = []
            print("inference_args:", inference_args)
        while has_unfinished_seqs():
            # print(count)
            # count = count + 1
            # print("vllm_waiting1:", vllm_waiting)
            scheduler_outputs = _schedule()
            # Create input data structures.
            seq_group_metadata_list: List[SequenceGroupMetadata] = []
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                seq_data: Dict[int, List[SequenceData]] = {}
                block_tables: Dict[int, List[int]] = {}
                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                    seq_id = seq.seq_id
                    seq_data[seq_id] = seq.data
                    block_tables[seq_id] = block_manager.get_block_table(seq)

                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=scheduler_outputs.prompt_run,
                    seq_data=seq_data,
                    inferencer_args=inference_args,
                    block_tables=block_tables,
                )
                seq_group_metadata_list.append(seq_group_metadata)
            ignored = [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]
            if scheduler_outputs.is_empty():
                step_outputs = ignored
            else:
                step_outputs = _run_workers(workers,
                                      "execute_model",
                                      seq_group_metadata_list=seq_group_metadata_list,
                                      blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                                      blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                                      blocks_to_copy=scheduler_outputs.blocks_to_copy, )
                if step_outputs == {}:
                    break
                
                step_outputs = _process_model_outputs(step_outputs, scheduler_outputs) + ignored
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
            # print("waiting:", len(vllm_waiting), "running:", len(vllm_running), "swapped:", len(vllm_swapped))
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        #print(outputs)


        # process input end

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        # outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            log_prob = output.outputs[0].logprobs
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Log_prob:{log_prob!r}")

    else:
        print("inference_args:", inference_args)
        for prompt in prompts:
            inputs = model.encode(prompt, return_tensors="pt").to(device=args_local_rank)
            outputs = model.inference(
                inputs,
                max_new_tokens=50,
                temperature=0,
                #num_beams=self.inferencer_args.num_beams,
                top_p=1.0,
                repetition_penalty=1.0,
                do_sample=False,
            )
            generated_text = model.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            # last_token_logits = model(inputs, return_dict=True).logits[0]
            # probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
            # log_prob = torch.log(probs)
            # print(log_prob[0][:50])

def has_unfinished_seqs() -> bool:
    return vllm_waiting or vllm_running or vllm_swapped

def _schedule() -> SchedulerOutputs:
    global vllm_swapped, vllm_waiting, vllm_running
    # Blocks that need to be swaped or copied before model execution.
    blocks_to_swap_in: Dict[int, int] = {}
    blocks_to_swap_out: Dict[int, int] = {}
    blocks_to_copy: Dict[int, List[int]] = {}

    # Fix the current time.
    now = time.monotonic()

    # Join waiting sequences if possible.
    if not vllm_swapped:
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in vllm_running)
        num_batched_tokens = 0
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while vllm_waiting:
            seq_group = vllm_waiting[0]

            assert seq_group.num_seqs() == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            if num_prompt_tokens > prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                vllm_waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            if not block_manager.can_allocate(seq_group):
                break

            # If the number of batched tokens exceeds the limit, stop.
            if (num_batched_tokens + num_prompt_tokens >
                    scheduler_config.max_num_batched_tokens):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    scheduler_config.max_num_seqs):
                break

            seq_group = vllm_waiting.pop(0)
            # status1 = seq_group.get_seqs()[0].status
            # print("seq_group1:", status1)
            # print("vllm_waiting2:", vllm_waiting)
            _allocate(seq_group)
            # status2 = seq_group.get_seqs()[0].status
            # print("seq_group2:", status2)
            vllm_running.append(seq_group)
            # print("vllm_running1:", vllm_running)
            num_batched_tokens += num_prompt_tokens
            num_curr_seqs += num_new_seqs
            scheduled.append(seq_group)

        if scheduled or ignored_seq_groups:
            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=True,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=ignored_seq_groups,
            )
            return scheduler_outputs

    # NOTE(woosuk): Preemption happens only when there is no available slot
    # to keep all the sequence groups in the RUNNING state.
    # In this case, the policy is responsible for deciding which sequence
    # groups to preempt.
    vllm_running = policy.sort_by_priority(now, vllm_running)

    # Reserve new token slots for the running sequence groups.
    running: List[SequenceGroup] = []
    preempted: List[SequenceGroup] = []
    while vllm_running:
        # print("vllm_running2:", vllm_running)
        seq_group = vllm_running.pop(0)
        # status3 = seq_group.get_seqs()[0].status
        # print("seq_group3:", status3)
        # print("vllm_running3:", vllm_running)
        while not block_manager.can_append_slot(seq_group):
            if vllm_running:
                # Preempt the lowest-priority sequence groups.
                victim_seq_group = vllm_running.pop(-1)
                _preempt(victim_seq_group, blocks_to_swap_out)
                preempted.append(victim_seq_group)
            else:
                # No other sequence groups can be preempted.
                # Preempt the current sequence group.
                _preempt(seq_group, blocks_to_swap_out)
                preempted.append(seq_group)
                break
        else:
            # Append new slots to the sequence group.
            _append_slot(seq_group, blocks_to_copy)
            running.append(seq_group)
            # status4 = seq_group.get_seqs()[0].status
            # print("seq_group4:", status4)
    vllm_running = running
    # print("vllm_running4:", vllm_running)

    # Swap in the sequence groups in the SWAPPED state if possible.
    vllm_swapped = policy.sort_by_priority(now, vllm_swapped)
    if not preempted:
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in vllm_running)
        # print("vllm_swapped1:", vllm_swapped)
        while vllm_swapped:
            seq_group = vllm_swapped[0]
            # If the sequence group cannot be swapped in, stop.
            if not block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    scheduler_config.max_num_seqs):
                break

            seq_group = vllm_swapped.pop(0)
            _swap_in(seq_group, blocks_to_swap_in)
            _append_slot(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            vllm_running.append(seq_group)

    # Each sequence in the generation phase only takes one token slot.
    # Therefore, the number of batched tokens is equal to the number of
    # sequences in the RUNNING state.
    num_batched_tokens = sum(
        seq_group.num_seqs(status=SequenceStatus.RUNNING)
        for seq_group in vllm_running)
    # print("num_batched_tokens:",num_batched_tokens)
    scheduler_outputs = SchedulerOutputs(
        scheduled_seq_groups=vllm_running,
        prompt_run=False,
        num_batched_tokens=num_batched_tokens,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        ignored_seq_groups=[],
    )
    return scheduler_outputs

def _allocate(seq_group: SequenceGroup) -> None:
    block_manager.allocate(seq_group)
    for seq in seq_group.get_seqs():
        seq.status = SequenceStatus.RUNNING

def _append_slot(
    seq_group: SequenceGroup,
    blocks_to_copy: Dict[int, List[int]],
) -> None:
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        ret = block_manager.append_slot(seq)
        if ret is not None:
            src_block, dst_block = ret
            if src_block in blocks_to_copy:
                blocks_to_copy[src_block].append(dst_block)
            else:
                blocks_to_copy[src_block] = [dst_block]

def _preempt(
    seq_group: SequenceGroup,
    blocks_to_swap_out: Dict[int, int],
    preemption_mode: Optional[PreemptionMode] = None,
) -> None:
    # If preemption mode is not specified, we determine the mode as follows:
    # We use recomputation by default since it incurs lower overhead than
    # swapping. However, when the sequence group has multiple sequences
    # (e.g., beam search), recomputation is not currently supported. In
    # such a case, we use swapping instead.
    # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
    # As swapped sequences are prioritized over waiting sequences,
    # sequence groups with multiple sequences are implicitly prioritized
    # over sequence groups with a single sequence.
    # TODO(woosuk): Support recomputation for sequence groups with multiple
    # sequences. This may require a more sophisticated CUDA kernel.
    if preemption_mode is None:
        if seq_group.get_max_num_running_seqs() == 1:
            preemption_mode = PreemptionMode.RECOMPUTE
        else:
            preemption_mode = PreemptionMode.SWAP
    if preemption_mode == PreemptionMode.RECOMPUTE:
        _preempt_by_recompute(seq_group)
    elif preemption_mode == PreemptionMode.SWAP:
        _preempt_by_swap(seq_group, blocks_to_swap_out)
    else:
        assert False, "Invalid preemption mode."

def _preempt_by_recompute(
    seq_group: SequenceGroup,
) -> None:
    seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    assert len(seqs) == 1
    for seq in seqs:
        seq.status = SequenceStatus.WAITING
        block_manager.free(seq)
    # NOTE: For FCFS, we insert the preempted sequence group to the front
    # of the waiting queue.
    vllm_waiting.insert(0, seq_group)

def _preempt_by_swap(

    seq_group: SequenceGroup,
    blocks_to_swap_out: Dict[int, int],
) -> None:
    _swap_out(seq_group, blocks_to_swap_out)
    vllm_swapped.append(seq_group)

def _swap_in(
    seq_group: SequenceGroup,
    blocks_to_swap_in: Dict[int, int],
) -> None:
    mapping = block_manager.swap_in(seq_group)
    blocks_to_swap_in.update(mapping)
    for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
        seq.status = SequenceStatus.RUNNING

def _swap_out(
    seq_group: SequenceGroup,
    blocks_to_swap_out: Dict[int, int],
) -> None:
    if not block_manager.can_swap_out(seq_group):
        # FIXME(woosuk): Abort the sequence group instead of aborting the
        # entire engine.
        raise RuntimeError(
            "Aborted due to the lack of CPU swap space. Please increase "
            "the swap space to avoid this error.")
    mapping = block_manager.swap_out(seq_group)
    blocks_to_swap_out.update(mapping)
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        seq.status = SequenceStatus.SWAPPED

@cached_property
def sampling_type(inferencer_args) -> SamplingType:
    if inferencer_args.use_beam_search:
        return SamplingType.BEAM
    if inferencer_args.temperature < _SAMPLING_EPS:
        return SamplingType.GREEDY
    return SamplingType.RANDOM


def _process_model_outputs(
        output: SamplerOutput,
        scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
    # Update the scheduled sequence groups with the model outputs.
    scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
    for seq_group, outputs in zip(scheduled_seq_groups, output):
        _process_sequence_group_outputs(seq_group, outputs)

    # Free the finished sequence groups.
    scheduler.free_finished_seq_groups()

    # Create the outputs.
    request_outputs: List[RequestOutput] = []
    for seq_group in (scheduled_seq_groups +
                        scheduler_outputs.ignored_seq_groups):
        request_output = RequestOutput.from_seq_group(seq_group)
        request_outputs.append(request_output)

    # if log_stats:
    #     # Log the system stats.
    #     _log_system_stats(scheduler_outputs.prompt_run,
    #                             scheduler_outputs.num_batched_tokens)
    return request_outputs

def _process_sequence_group_outputs(seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutputs) -> None:
    # Process prompt logprobs
    prompt_logprobs = outputs.prompt_logprobs
    if prompt_logprobs is not None:
        seq_group.prompt_logprobs = prompt_logprobs

    # Process samples
    samples = outputs.samples
    parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    existing_finished_seqs = seq_group.get_finished_seqs()
    parent_child_dict = {
        parent_seq.seq_id: []
        for parent_seq in parent_seqs
    }
    for sample in samples:
        parent_child_dict[sample.parent_seq_id].append(sample)
    # List of (child, parent)
    child_seqs: List[Tuple[Sequence, Sequence]] = []

    # Process the child samples for each parent sequence
    for parent in parent_seqs:
        child_samples: List[SequenceOutputs] = parent_child_dict[
            parent.seq_id]
        if len(child_samples) == 0:
            # This parent sequence has no children samples. Remove
            # the parent sequence from the sequence group since it will
            # not be used in the future iterations.
            parent.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(parent.seq_id)
            scheduler.free_seq(parent)
            continue
        # Fork the parent sequence if there are multiple child samples.
        for child_sample in child_samples[:-1]:
            new_child_seq_id = next(seq_counter)
            child = parent.fork(new_child_seq_id)
            child.append_token_id(child_sample.output_token,
                                    child_sample.logprobs)
            child_seqs.append((child, parent))
        # Continue the parent sequence for the last child sample.
        # We reuse the parent sequence here to reduce redundant memory
        # copies, especially when using non-beam search sampling methods.
        last_child_sample = child_samples[-1]
        parent.append_token_id(last_child_sample.output_token,
                                last_child_sample.logprobs)
        child_seqs.append((parent, parent))

    for seq, _ in child_seqs:
        _decode_sequence(seq, seq_group.inferencer_args)
        _check_stop(seq, seq_group.inferencer_args)

    # Non-beam search case
    if not seq_group.inferencer_args.use_beam_search:
        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        # NOTE: we need to fork the new sequences before freeing the
        # old sequences.
        for seq, parent in child_seqs:
            if seq is parent and seq.is_finished():
                scheduler.free_seq(seq)
        return

    # Beam search case
    # Select the child sequences to keep in the sequence group.
    selected_child_seqs = []
    unselected_child_seqs = []
    beam_width = seq_group.inferencer_args.num_beams
    length_penalty = seq_group.inferencer_args.length_penalty

    # Select the newly finished sequences with the highest scores
    # to replace existing finished sequences.
    # Tuple of (seq, parent, is_new)
    existing_finished_seqs = [(seq, None, False)
                                for seq in existing_finished_seqs]
    new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                            if seq.is_finished()]
    all_finished_seqs = existing_finished_seqs + new_finished_seqs
    # Sort the finished sequences by their scores.
    all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id),
                            reverse=True)
    for seq, parent, is_new in all_finished_seqs[:beam_width]:
        if is_new:
            # A newly generated child sequence finishes and has a high
            # score, so we will add it into the sequence group.
            selected_child_seqs.append((seq, parent))
    for seq, parent, is_new in all_finished_seqs[beam_width:]:
        if is_new:
            # A newly generated child sequence finishes but has a low
            # score, so we will not add it into the sequence group.
            # Additionally, if this sequence is a continuation of a
            # parent sequence, we will need remove the parent sequence
            # from the sequence group.
            unselected_child_seqs.append((seq, parent))
        else:
            # An existing finished sequence has a low score, so we will
            # remove it from the sequence group.
            seq_group.remove(seq.seq_id)

    # select the top beam_width sequences from the running
    # sequences for the next iteration to continue the beam
    # search.
    running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                            if not seq.is_finished()]
    # Sort the running sequences by their scores.
    running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id),
                            reverse=True)

    # Check if we can stop the beam search.
    if len(running_child_seqs) == 0:
        # No running sequences, stop the beam search.
        stop_beam_search = True
    elif len(all_finished_seqs) < beam_width:
        # Not enough finished sequences, continue the beam search.
        stop_beam_search = False
    else:
        # Check the early stopping criteria
        best_running_seq = running_child_seqs[0][0]
        current_worst_seq = all_finished_seqs[beam_width - 1][0]
        stop_beam_search = _check_beam_search_early_stopping(
            seq_group.inferencer_args.early_stopping,
            seq_group.inferencer_args, best_running_seq, current_worst_seq)

    if stop_beam_search:
        # Stop the beam search and remove all the running sequences from
        # the sequence group.
        unselected_child_seqs.extend(running_child_seqs)
    else:
        # Continue the beam search and select the top beam_width sequences
        # to continue the beam search.
        selected_child_seqs.extend(running_child_seqs[:beam_width])
        # The remaining running sequences will not be used in the next
        # iteration. Again, if these sequences are continuations of
        # parent sequences, we will need to remove the parent sequences
        # from the sequence group.
        unselected_child_seqs.extend(running_child_seqs[beam_width:])

    # For newly created child sequences, add them to the sequence group
    # and fork them in block manager if they are not finished.
    for seq, parent in selected_child_seqs:
        if seq is not parent:
            seq_group.add(seq)
            if not seq.is_finished():
                scheduler.fork_seq(parent, seq)

    # Free the finished and selected parent sequences' memory in block
    # manager. Keep them in the sequence group as candidate output.
    for seq, parent in selected_child_seqs:
        if seq is parent and seq.is_finished():
            scheduler.free_seq(seq)

    # Remove the unselected parent sequences from the sequence group and
    # free their memory in block manager.
    for seq, parent in unselected_child_seqs:
        if seq is parent:
            # Remove the parent sequence if it is not selected for next
            # iteration
            seq_group.remove(seq.seq_id)
            scheduler.free_seq(seq)

def _check_stop(seq: Sequence,
                    infer_args: InferencerArguments) -> None:
    """Stop the finished sequences."""
    # for stop_str in infer_args.stop:
    #     if seq.output_text.endswith(stop_str):
    #         # Truncate the output text so that the stop string is
    #         # not included in the output.
    #         seq.output_text = seq.output_text[:-len(stop_str)]
    #         seq.status = SequenceStatus.FINISHED_STOPPED
    #         return
    # if seq.get_last_token_id() in infer_args.stop_token_ids:
    #     seq.status = SequenceStatus.FINISHED_STOPPED
    #     return

    # Check if the sequence has reached max_model_len.
    if seq.get_len() > scheduler_config.max_model_len:
        seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
        return

    # Check if the sequence has reached max_tokens.
    if seq.get_output_len() == infer_args.max_new_tokens:
        seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
        return

    # Check if the sequence has generated the EOS token.
    if ((not infer_args.ignore_eos)
            and seq.get_last_token_id() == tokenizer.eos_token_id):
        seq.status = SequenceStatus.FINISHED_STOPPED
        return

def _decode_sequence(seq: Sequence, prms: InferencerArguments) -> None:
    """Decodes the new token for a sequence."""
    (new_tokens, new_output_text, prefix_offset,
        read_offset) = detokenize_incrementally(
            tokenizer,
            all_input_ids=seq.get_token_ids(),
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=prms.skip_special_tokens,
            spaces_between_special_tokens=prms.spaces_between_special_tokens,
        )
    if seq.tokens is None:
        seq.tokens = new_tokens
    else:
        seq.tokens.extend(new_tokens)
    seq.prefix_offset = prefix_offset
    seq.read_offset = read_offset
    seq.output_text += new_output_text

def _check_beam_search_early_stopping(
    early_stopping: Union[bool, str],
    infer_args: InferencerArguments,
    best_running_seq: Sequence,
    current_worst_seq: Sequence,
) -> bool:
    assert infer_args.use_beam_search
    length_penalty = infer_args.length_penalty
    if early_stopping is True:
        return True

    current_worst_score = (current_worst_seq.get_beam_search_score(
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id))
    if early_stopping is False:
        highest_attainable_score = (best_running_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=tokenizer.eos_token_id))
    else:
        assert early_stopping == "never"
        if length_penalty > 0.0:
            # If length_penalty > 0.0, beam search will prefer longer
            # sequences. The highest attainable score calculation is
            # based on the longest possible sequence length in this case.
            max_possible_length = max(
                best_running_seq.get_prompt_len() +
                infer_args.max_new_tokens,
                scheduler_config.max_model_len)
            highest_attainable_score = (
                best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                    seq_len=max_possible_length))
        else:
            # Otherwise, beam search will prefer shorter sequences. The
            # highest attainable score calculation is based on the current
            # sequence length.
            highest_attainable_score = (
                best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=tokenizer.eos_token_id))
    return current_worst_score >= highest_attainable_score


if __name__ == "__main__":
    main()