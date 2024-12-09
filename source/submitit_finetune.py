# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import submitit
from utils import arg_util


def submitit_parse_args():
    trainer_parser = arg_util.get_args_parser()

    parser = argparse.ArgumentParser("Submitit for SpaT-Spark Finetune", parents=[trainer_parser], add_help=False)
    # parser = argparse.ArgumentParser("Submitit for Spark pretrain")
    parser.add_argument("--partition", default="gpua100", type=str, help="Partition where to submit")
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=5760, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument('--mail_user', default='h.li2@uu.nl', type=str)
    parser.add_argument('--mail_type', default='ALL', type=str)
    parser.add_argument("--dependency", default=None, type=str, help="Slurm job id to depend on")
    parser.add_argument("--cpu_per_task", default=8, type=int, help="Number of CPUs per task")
    parser.add_argument("--mem_per_task", default=20, type=int, help="Memory per task in GB")
    parser.add_argument("--time", default="2-00:00:00", type=str, help="Time limit for the job in slurm format")
    return parser.parse_args()


def get_shared_folder(args) -> Path:
    # user = os.getenv("USER")
    # if Path("/checkpoint/").is_dir():
    #     p = Path(f"/checkpoint/{user}/experiments")
    #     p.mkdir(exist_ok=True)
    #     return p
    # raise RuntimeError("No shared folder available")
    if args.partition == "gpu":
        return Path("./experiments/")
    else:
        raise ValueError(f"Unknown partition {args.partition}, cannot determine shared folder.")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import finetune_main as trainer

        self._setup_gpu_args()
        trainer.main_finetune(args=self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = submitit_parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    if args.comment:
        kwargs['slurm_comment'] = args.comment
    if args.mail_user:
        kwargs['slurm_mail_user'] = args.mail_user
    if args.mail_type:
        kwargs['slurm_mail_type'] = args.mail_type
    if args.dependency:
        kwargs['slurm_dependency'] = args.dependency
    if args.time:
        kwargs['slurm_time'] = args.time

    executor.update_parameters(
        mem_gb=args.mem_per_task * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpu_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="SpaT-Spark_FT" if args.exp_name == "" else "SpaT-SparK_FT_"+args.exp_name)

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    print(args.exp_name)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()
