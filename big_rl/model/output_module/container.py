import torch


class OutputModuleContainer(torch.nn.Module):
    def __init__(self, output_modules: torch.nn.ModuleDict):
        super().__init__()

        self.output_modules = output_modules

    def forward(self, key, value):
        output = {}
        for k,module in self.output_modules.items():
            output[k] = module(key, value)['output']

        return output
