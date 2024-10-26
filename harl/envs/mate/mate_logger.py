from harl.common.base_logger import BaseLogger
import numpy as np


class MateLogger(BaseLogger):
    def get_task_name(self):
        return "mate"

    def eval_init(self):
        super().eval_init()
        self.rate_list = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.rate_list.append(0)
        self.final_rate_list = []

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        super().eval_per_step(eval_data)
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            if self.eval_infos[eval_i].get("coverage_rate"):
                self.rate_list[eval_i] = self.eval_infos.get("coverage_rate")

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self.final_rate_list.append(self.rate_list[tid])

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        mean_coverage = np.mean(self.final_rate_list)
        self.final_rate_list = []
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        print("Evaluation average episode coverage rate is {}.\n".format(mean_coverage))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew, mean_coverage])) + "\n"
        )
        self.log_file.flush()