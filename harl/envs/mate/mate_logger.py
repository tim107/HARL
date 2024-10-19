from harl.common.base_logger import BaseLogger


class MateLogger(BaseLogger):
    def get_task_name(self):
        return "mate"
