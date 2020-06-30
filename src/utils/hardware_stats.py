import torch
import nvidia_smi
import psutil


class HardwareStats():
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def hardware_stats(self):
        """
        Returns a dict containing some hardware related stats
        """
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        return {"cpu": f"{str(psutil.cpu_percent())}%",
                "mem": f"{str(psutil.virtual_memory().percent)}%",
                "gpu": f"{str(res.gpu)}%",
                "gpu_mem": f"{str(res.memory)}%"}
