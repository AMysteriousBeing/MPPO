from multiprocessing import Process, shared_memory
from utils import load_actions, data_augmentation
import torch
import uuid
import time
import json
import os
import numpy as np
import random
from environment import MazeEnv2
from agent import MazeAgentBasic
from model import MazeNet2
from utils import CustomLogger


LOG_FORMAT = (
    "%(levelname) -8s %(asctime)s %(name) -25s %(funcName) "
    "-25s %(lineno) -5d: %(message)s"
)


class ActorMazeSplit(Process):
    """
    Seperate true trajectory actors from sampling actors for better data control
    Once guided-actors iterate through all data a fixed numer of times,
    guided-actors fall back to sampling-actors
    Simplified ActorMahjongSplit from ActorMahjongSplit. All experimental code & scaffoldings are removed.
    """

    def __init__(
        self,
        model_pool_clt,
        replay_buffer_act,
        config,
        actor_name="",
        eval_actor=False,
        guide_trajectory=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.debug = config.get("debug", False)
        self.oracle = config.get("oracle", False)
        if self.debug:
            self.model_pool_clt = None
        else:
            self.model_pool_clt = model_pool_clt
        self.eval = eval_actor
        self.config = config
        self.early_kill = self.config["early_kill"]

        # trajectory sampling and augmentation scheme
        self.guide_trajectory = guide_trajectory
        self.self_play = self.config["self_play"]
        self.init_hands = self.config["init_hands"]
        self.enable_augmentation = self.config["augmentation"]

        self.replay_buffer_act = replay_buffer_act
        self.act_id = uuid.uuid4().hex
        self.name = actor_name
        rand_seed = int(actor_name.split("-")[-1]) * self.config["seed"]
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        # need write to config
        self.data_dir = self.config["path_to_data"]
        self.eval_data_dir = self.config["path_to_eval_data"]
        self.available_games = os.listdir(self.data_dir)
        self.available_games.sort()
        self.available_eval_games = os.listdir(self.eval_data_dir)

        self.additional_log_dir = os.path.join(
            "logs", self.config["log_dir"], "additional_logs"
        )
        if not os.path.exists(self.additional_log_dir):
            os.makedirs(self.additional_log_dir)

        self.logger = CustomLogger(
            os.path.join(self.additional_log_dir, "{}.txt".format(self.name))
        )

        if self.debug:
            self.actor_routine_status_list = [0, 0, 0]
        else:
            # get run status from actor routine
            self.shm = shared_memory.SharedMemory(name="actor_routine_sm4")
            self.actor_routine_status_list = np.ndarray(
                3, dtype="int64", buffer=self.shm.buf
            )
        self.model_tag_id = -1

    def run(self):
        torch.set_num_threads(1)
        game_counter = 0
        model = MazeNet2()
        model.eval()
        self.logger.info("Starting {}".format(self.name))
        while not self.actor_routine_status_list[1]:
            # load balancing
            time.sleep(random.uniform(0, 0.5))

            if self.actor_routine_status_list[2]:
                continue
            # get model
            if not (self.debug):
                try:
                    returned_state_dict = self.model_pool_clt.load_model(
                        self.model_pool_clt.get_latest_model()
                    )
                except Exception as e:
                    self.logger.info("Encountered Error: {}".format(e))
                    self.logger.info("Recovering from error with retry")
                    time.sleep(0.2)
                    continue

                state_dict = returned_state_dict["state_dict"]
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                    state_dict, "module."
                )
                self.model_tag_id = returned_state_dict["tag"]
                model.load_state_dict(state_dict)

            if not self.self_play or self.guide_trajectory:
                # initialize less-randomized environment
                if not self.eval:
                    # randomly sample from available games
                    sampling_id = random.choice(range(self.init_hands))

                    file_name = self.available_games[sampling_id]
                    # load game data from file
                    file_path = os.path.join(self.data_dir, file_name)
                else:
                    # get eval file path
                    file_name = random.choice(self.available_eval_games)
                    file_path = os.path.join(self.eval_data_dir, file_name)

                with open(file_path, "r") as f:
                    recorded_data = json.load(f)

                config = {"agent_type": MazeAgentBasic, "oracle": self.oracle}
                env = MazeEnv2(config)
                episode_data = {
                    "player_1": {
                        "state": {"cnn_obs": [], "dense_obs": [], "action_mask": []},
                        "action": [],
                        "reward": [],
                        "value": [],
                        "info": [],
                    }
                }

                obs, reward, done = env.reset(init_seed=recorded_data["seed"])
                # load history actions from augmented game setup
                action_history = recorded_data["action_list"]
                action_pointer = 0
            else:
                # initialize environment as normal training sessions
                config = {"agent_type": MazeAgentBasic, "oracle": self.oracle}
                env = MazeEnv2(config)
                episode_data = {
                    "player_1": {
                        "state": {"cnn_obs": [], "dense_obs": [], "action_mask": []},
                        "action": [],
                        "reward": [],
                        "value": [],
                        "info": [],
                    }
                }
                obs, reward, done = env.reset()

            step_counter = 6
            game_length = 0
            game_counter += 1
            sampled_steps = 0
            correct_action_flag = (not self.eval) and self.guide_trajectory
            model.train(False)  # Batch Norm inference mode
            while not done:
                step_counter += 1
                # each player take action
                actions = {}
                values = {}
                # infer with network
                agent_data = episode_data["player_1"]
                for k, v in obs.items():
                    agent_data["state"][k].append(v)
                obs2tensor = {}
                for k, v in obs.items():
                    obs2tensor[k] = v.unsqueeze(0)

                with torch.no_grad():

                    logits, value = model(obs2tensor)
                    action_dist = torch.distributions.Categorical(logits=logits)
                    action = action_dist.sample().item()
                    value = value.item()
                # use correct action
                if correct_action_flag:
                    # calculate correct action
                    action = action_history[action_pointer]
                    action_pointer += 1
                actions["player_1"] = action
                values["player_1"] = value
                agent_data["action"].append(actions["player_1"])
                agent_data["value"].append(values["player_1"])
                agent_data["info"].append(
                    [
                        1,
                        0,
                        self.model_tag_id,
                        0,
                    ]
                )

                # interact with env
                next_obs, rewards, done = env.step(action)
                episode_data["player_1"]["reward"].append(rewards)
                obs = next_obs
                game_length += 1

            # gather necessary data
            # game_length, sampled_steps(total), win/loss

            win = rewards > 0
            if win:
                winner_id = 1
            else:
                winner_id = -1

            # postprocessing episode data for each agent
            sampled_steps = 0
            for agent_name, agent_data in episode_data.items():
                if len(agent_data["action"]) < len(agent_data["reward"]):
                    print(agent_data["action"], len(agent_data["reward"]))
                    agent_data["reward"].pop(0)
                if len(agent_data["state"]["action_mask"]) > 0:
                    cnn_obs = np.stack(agent_data["state"]["cnn_obs"])
                    dense_obs = np.stack(agent_data["state"]["dense_obs"])
                    mask = np.stack(agent_data["state"]["action_mask"])
                    actions = np.array(agent_data["action"], dtype=np.int64)
                    rewards = np.array(agent_data["reward"], dtype=np.float32)
                    values = np.array(agent_data["value"], dtype=np.float32)
                    next_values = np.array(
                        agent_data["value"][1:] + [0], dtype=np.float32
                    )

                    td_target = rewards + next_values * self.config["gamma"]
                    td_delta = td_target - values
                    advs = []
                    adv = 0
                    for delta in td_delta[::-1]:
                        adv = self.config["gamma"] * self.config["lambda"] * adv + delta
                        advs.append(adv)  # GAE
                    advs.reverse()
                    advantages = np.array(advs, dtype=np.float32)
                    info_list = np.array(agent_data["info"]).reshape(-1, 4)
                    if not self.debug and not self.eval:
                        # send samples to replay_buffer (per agent)
                        self.replay_buffer_act.push(
                            {
                                "state": {
                                    "cnn_obs": cnn_obs,
                                    "dense_obs": dense_obs,
                                    "action_mask": mask,
                                },
                                "action": actions,
                                "adv": advantages,
                                "target": td_target,
                                "info": info_list,  # player_id, winner_id, tag, augmentation scheme id
                            }
                        )
                    if self.debug:
                        print(
                            {
                                "state": obs,
                                "action": actions,
                                "adv": advantages,
                                "target": td_target,
                                "info": info_list,  # player_id, winner_id, tag, augmentation scheme id
                            }
                        )
                    sampled_steps += len(actions)
            if not self.debug:
                self.replay_buffer_act.push(
                    {
                        "log": {
                            "win": win,
                            "eval": self.eval,
                            "guided": self.guide_trajectory,
                            "sampled_step": sampled_steps,
                            "game_length": game_length,
                            "reward": sum(agent_data["reward"]),
                            "fan_list": ["NULL"],
                        }
                    }
                )

            # conpensate for faster iterations
            # if self.guide_trajectory:
            #     time.sleep(random.uniform(0.15, 0.3))
            if self.debug:
                break
                time.sleep(1)
                if game_counter > 3:
                    break
        if not (self.debug):
            self.shm.close()
