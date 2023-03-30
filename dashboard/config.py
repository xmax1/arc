from pathlib import Path
from typing import Callable, List, Tuple
import os
import torch
from pydantic import BaseModel

class UserConfig(BaseModel):
    name: str
    secret: str

class Pyfig(BaseModel):
    user: UserConfig
    project: str = None
    env: str = None
    run_name: Path = None
    exp_name: str = None
    exp_id: str = None
    group_exp: bool = False
    lo_ve_path: str = None
    mode: str = None
    multimode: str = None
    debug: bool = False
    seed: int = 0
    dtype_str: str = 'float32'

    run_id: str = None
    submit: bool = False
    dtype: torch.dtype = None
    device: str = None

    n_log_metric: int = 100
    n_log_state: int = 4
    is_logging_process: bool = False

    cudnn_benchmark: bool = True

    zweep: str = ''

    n_default_step: int = 10
    n_train_step: int = 0
    n_pre_step: int = 0
    n_eval_step: int = 0
    n_opt_hypam_step: int = 0
    n_max_mem_step: int = 0

    data_tag: str = 'data'
    max_mem_alloc_tag: str = 'max_mem_alloc'
    opt_obj_all_tag: str = 'opt_obj_all'
    opt_obj_tag: str = 'opt_obj'

    pre_tag: str = 'pre'
    train_tag: str = 'train'
    eval_tag: str = 'eval'
    opt_hypam_tag: str = 'opt_hypam'

    v_cpu_d_tag: str = 'v_cpu_d'
    c_update_tag: str = 'c_update'

    lo_ve_path_tag: str = 'lo_ve_path'
    gather_tag: str = 'gather'
    mean_tag: str = 'mean'

    ignore_f: List[str] = ['commit', 'pull', 'backward']
    ignore_p: List[str] = ['parameters', 'scf', 'tag', 'mode_c']
    ignore: List[str] = ['ignore', 'ignore_f', 'ignore_c'] + ignore_f + ignore_p
    ignore += ['d', 'cmd', 'sub_ins', 'd_flat', 'repo', 'name', 'base_d', 'c_init', 'p']
    base_d: dict = None
    c_init: dict = None
    run_debug_c: bool = False
    run_sweep: bool = False

    @property
    def n_step(self):
        n_step = {
            'train': self.n_train_step,
            'pre': self.n_pre_step,
            'eval': self.n_eval_step,
            'opt_hypam': self.n_opt_hypam_step,
            'max_mem': self.n_max_mem_step
        }.get(self.mode)
        if not n_step:
            n_step = self.n_default_step
        return n_step

    @property
    def dtype(self):
        return {'float64': torch.float64, 'float32': torch.float32, 'cpu': 'cpu'}[self.dtype_str]

    @dtype.setter
    def dtype(self, val):
        if val is not None:
            self.dtype_str = str(val).split('.')[-1]

class Model(BaseModel):
    compile_ts: bool = False
    compile_func: bool = False
    optimise_ts: bool = False
    optimise_aot: bool = False
    with_sign: bool = False
    functional: bool = True

    terms_s_emb: List[str] = ['ra', 'ra_len']
    terms_p_emb: List[str] = ['rr', 'rr_len']
    ke_method: str = 'grad_grad'
    n_sv: int = 32
    n_pv: int = 32
    n_fb: int = 3
    n_det: int = 4
    n_final_out: int = 1

    @property
    def n_fbv(self) -> int:
        return self.n_sv * 3 + self.n_pv * 2


class Opt(BaseModel):
    available_opt: List[str] = ['AdaHessian', 'RAdam', 'Apollo', 'AdaBelief', 'LBFGS', 'Adam', 'AdamW']
    opt_name: str = 'AdamW'
    lr: float = 1e-3
    init_lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    beta: float = 0.9
    warm_up: int = 100
    eps: float = 1e-4
    weight_decay: float = 0.0
    hessian_power: float = 1.0


class Scheduler(BaseModel):
    sch_name: str = 'ExponentialLR'
    sch_max_lr: float = 0.01
    sch_epochs: int = 1
    sch_gamma: float = 0.9999
    sch_verbose: bool = False


class Sweep(BaseModel):
    sweep_name: str = 'study'
    n_trials: int = 20
    parameters: Dict[str, Any] = {
        'opt_name': {
            'values': ['AdaHessian', 'RAdam'],
            'dtype': str
        },
        'hessian_power': {
            'values': [0.5, 0.75, 1.],
            'dtype': float,
            'condition': ['AdaHessian']
        },
        'weight_decay': {
            'domain': (0.0001, 1.),
            'dtype': float,
            'condition': ['AdaHessian']
        },
        'lr': {
            'domain': (0.0001, 1.),
            'log': True,
            'dtype': float
        },
    }


class DistBase(BaseModel):
    dist_name: str = 'Base'
    n_launch: int = 1

    @property
    def n_worker(self) -> int:
        return self.p.resource.n_gpu

    ready: bool = True
    sync_every: int = 1
    rank_env_name: str = 'RANK'

    @property
    def rank(self) -> int:
        return int(os.environ.get(self.rank_env_name, '-1'))

    @property
    def head(self) -> bool:
        return self.rank == 0

    gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
    dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])
    pid: 			int = property(lambda _: _.rank)
  
  	_gpu_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'


from pydantic import BaseModel, validator


class Resource(BaseModel):
    n_gpu: int = 0
    n_node: int = 1
    n_thread_per_process: int = 1

    def cluster_submit(self, job: dict):
        return job

    def device_log_path(self, rank: int = 0):
        if not rank:
            return self.p.paths.exp_dir / f"{rank}_device.log"
        else:
            return self.p.paths.cluster_dir / f"{rank}_device.log"


class Niflheim(Resource):
    n_gpu: int = 1

    @validator("n_node", pre=True, always=True)
    def calculate_n_node(cls, v, values):
        return int(math.ceil(values.get("n_gpu", 0) / 10))

    @validator("n_thread_per_process", pre=True, always=True)
    def calculate_n_thread_per_process(cls, v, values):
        return values.get("slurm_c", {}).get("cpus_per_gpu", 8)

    architecture: str = "cuda"
    nifl_gpu_per_node: int = 10

    job_id: str = os.environ.get("SLURM_JOBID", "No SLURM_JOBID available.")  # slurm only

    _pci_id_cmd: str = "nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader"
    pci_id: str = "".join(run_cmds(self._pci_id_cmd, silent=True))

    n_device_env: str = "CUDA_VISIBLE_DEVICES"
    n_device: int = len(os.environ.get(self.n_device_env, "").replace(",", ""))

    class SlurmC(BaseModel):
        export: str = "ALL"
        cpus_per_gpu: int = 8  # 1 task 1 gpu 8 cpus per task
        partition: str = "sm3090"
        time: str = "0-00:10:00"  # D-HH:MM:SS
        nodes: str = ""
        gres: str = ""
        ntasks: int = 0
        job_name: str = ""
        output: str = ""
        error: str = ""

        @validator("nodes", pre=True, always=True)
        def calculate_nodes(cls, v, values):
            return str(values.get("p", {}).get("n_node", 1))

        @validator("gres", pre=True, always=True)
        def calculate_gres(cls, v, values):
            n_gpu = values.get("p", {}).get("n_gpu", 1)
            return f"gpu:RTX3090:{min(10, n_gpu)}"

        @validator("ntasks", pre=True, always=True)
        def calculate_ntasks(cls, v, values):
            nodes = int(values.get("nodes", 1))
            n_gpu = values.get("p", {}).get("n_gpu", 1)
            return n_gpu if nodes == 1 else nodes * 80

        @validator("job_name", pre=True, always=True)
        def calculate_job_name(cls, v, values):
            return values.get("p", {}).get("exp_name", "")

        @property
        def output(self):
            return self.p.p.paths.cluster_dir / 'o-%j.out'
    
        @property
        def error(self):
            return self.p.p.paths.cluster_dir / 'e-%j.err'

from typing import Optional
from pydantic import BaseModel

class Niflheim(BaseModel):
    n_gpu: int = 1
    architecture: str = 'cuda'
    nifl_gpu_per_node: int = 10
    
    @property
    def n_node(self) -> int:
        return int(math.ceil(self.n_gpu / self.nifl_gpu_per_node))

    @property
    def n_thread_per_process(self) -> int:
        return self.slurm_c.cpus_per_gpu

    @property
    def job_id(self) -> str:
        return os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.')  # slurm only

    _pci_id_cmd: str = 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'

    @property
    def pci_id(self) -> str:
        return ''.join(run_cmds(self._pci_id_cmd, silent=True))

    n_device_env: str = 'CUDA_VISIBLE_DEVICES'

    @property
    def n_device(self) -> int:
        return len(os.environ.get(self.n_device_env, '').replace(',', ''))

    class slurm_c(BaseModel):
        export: str = 'ALL'
        cpus_per_gpu: int = 8  # 1 task 1 gpu 8 cpus per task
        partition: str = 'sm3090'
        time: str = '0-00:10:00'  # D-HH:MM:SS
    
        @property
        def nodes(self) -> str:
            return str(self.p.n_node)  # (MIN-MAX)
    
        @property
        def gres(self) -> str:
            return 'gpu:RTX3090:' + str(min(10, self.p.n_gpu))
    
        @property
        def ntasks(self) -> int:
            return self.p.n_gpu if int(self.nodes) == 1 else int(self.nodes) * 80
    
        @property
        def job_name(self) -> str:
            return self.p.p.exp_name
  
      @property
      def output(self) -> str:
          return str(self.p.p.paths.cluster_dir/'o-%j.out')
  
      @property
      def error(self) -> str:
          return str(self.p.p.paths.cluster_dir/'e-%j.err')


