import torch


class Device:
    __availabled = (['cuda'] * int(torch.cuda.is_available())
                    + ['mps'] * int(torch.backends.mps.is_available())
                    + ['cpu', 'auto'])

    def __init__(self, device_name: str):
        self.__device_validate(device_name)
        self.__set_device(device_name)

    def __call__(self, device_name: str) -> torch.device:
        self.__device_validate(device_name)
        self.__set_device(device_name)
        return self._device

    def now(self) -> torch.device:
        return self._device

    def __device_validate(self, device_name: str):
        if device_name not in self.__availabled:
            raise ValueError('Device %s not available')

    def __set_device(self, device_name: str):
        if device_name == 'auto':
            self._device = self.__set_auto()
        self._device = torch.device(device_name)

    def __set_auto(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')


class DeviceManager:
    def __init__(self):
        self.device = Device('auto')

    def set(self, device_name: str):
        self.device(device_name)

    def get(self) -> torch.device:
        return self.device.now()
