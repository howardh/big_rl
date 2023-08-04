from collections import defaultdict
import torch

from big_rl.model.input_module.modules import IgnoredInput


class InputModuleContainer(torch.nn.Module):
    def __init__(self, input_modules: torch.nn.ModuleDict, strict: bool = False, input_mapping: list[tuple] = []):
        """
        Args:
            input_modules: A dictionary of input modules. Each module must have a forward method that takes a dictionary of inputs and returns a dictionary of outputs.
            strict: If True, then the input dictionary must contain a key for each input module. If False, then the input dictionary may contain only a subset of the input modules.
            input_mapping: A list of mappings from input module names to input keys. By default, the input module name is used as the input key.
                A single input key may be mapped to multiple input modules.
                A single input module may be mapped to multiple input keys, either by running the module once for each input key, or by combining the input keys into a single input dictionary and using them as keyword arguments to the module.
        """
        super().__init__()

        if strict:
            raise NotImplementedError()

        self.input_modules = input_modules
        self._input_mapping = input_mapping
        self._input_to_modules = defaultdict(set)
        for input_key, module_name in input_mapping:
            if module_name not in input_modules:
                raise ValueError(f"Unknown input module name: {module_name}")
            self._input_to_modules[input_key].add(module_name)
        for k in self.input_modules.keys():
            if k not in self._input_to_modules:
                self._input_to_modules[k].add(k)

    def forward(self, inputs):
        input_labels = []
        input_keys = []
        input_vals = []
        for k,v in inputs.items():
            module_names = self._input_to_modules[k]

            # Make sure the input is mapped to at least one module. If it is meant to be ignored, it should be explicitly ignored. Silent bugs are less likely this way.
            if len(module_names) == 0:
                raise ValueError(f"Unknown input key: {k}")

            for module_name in module_names:
                module = self.input_modules[module_name]

                if isinstance(module, IgnoredInput):
                    continue

                module_output = module(v)

                input_labels.append(module_name)
                input_keys.append(module_output['key'].unsqueeze(0))
                input_vals.append(module_output['value'].unsqueeze(0))

        return {
            'key': input_keys,
            'value': input_vals,
            'misc': {
                'input_labels': input_labels,
            }
        }
