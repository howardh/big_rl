import torch


class OutputModuleContainer(torch.nn.Module):
    def __init__(self, output_modules: torch.nn.ModuleDict):
        super().__init__()

        self.output_modules = output_modules

        if 'misc' in self.output_modules:
            raise ValueError('Output module name "misc" is reserved')

    def forward(self, key, value):
        output = {'misc': {}}
        for k,module in self.output_modules.items():
            o = module(key, value)
            output[k] = o['output']
            output['misc'][k] = o.get('misc', None)

        return output
