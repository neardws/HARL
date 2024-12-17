from harl.common.base_logger import BaseLogger


class VECLogger(BaseLogger):
    def get_task_name(self):
        return "VECEnv"