from multiprocessing import Process, shared_memory
from utils import load_actions, data_augmentation
import torch
import uuid
import time
import json
import os
import numpy as np
import random
from agent import (
    ZombieAgent,
    SearchAgentSlim,
    FeatureAgent2Adapted,
    FeatureAgent2Adapted2a,
)
from environment import MahjongGBEnv
from model import MahJongCNNNet6_LargeV2, MahJongNet64a
from utils import CustomLogger

LOG_FORMAT = (
    "%(levelname) -8s %(asctime)s %(name) -25s %(funcName) "
    "-25s %(lineno) -5d: %(message)s"
)
TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]


class ActorMahJongSplit(Process):
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
        self.debug = False  # switch for production or debug
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
        # need write to config
        self.data_dir = self.config["path_to_data"]
        self.eval_data_dir = self.config["path_to_eval_data"]
        self.available_games = os.listdir(self.data_dir)
        self.available_games.sort()
        self.available_eval_games = os.listdir(self.eval_data_dir)[31300:]

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
        model = MahJongCNNNet6_LargeV2()
        model.eval()
        self.logger.info("Starting {}".format(self.name))

        while not self.actor_routine_status_list[1]:
            # load balancing
            time.sleep(random.uniform(0, 0.3))

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

            if not self.self_play or self.guide_trajectory or self.eval:
                # initialize less-randomized environment
                winner_data_only = True
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
                historical_winner_id = recorded_data["winner_id"]
                # historical_winner_id = random.randint(0, 3)

                # select augmentation key given sampling setup
                if recorded_data["augmentable"] == True:
                    augmentation_key = random.randint(0, 11)
                else:
                    augmentation_key = 0

                for i in range(4):
                    self.config["agent_clz{}".format(i)] = (
                        FeatureAgent2Adapted
                        if i == historical_winner_id
                        else ZombieAgent
                    )
                env = MahjongGBEnv(self.config)
                episode_data = {
                    agent_name: {
                        "state": {"observation": [], "action_mask": []},
                        "action": [],
                        "reward": [],
                        "value": [],
                        "info": [],
                    }
                    for agent_name in env.agent_names
                }

                obs = env.reset(
                    recorded_data["wind"],
                    data_augmentation(recorded_data["initwall"], augmentation_key),
                )
                agents = env.return_agent_list()
                done = False
                # load history actions from augmented game setup
                action_history = load_actions(
                    recorded_data["history"], augmentation_key
                )
                action_pointer = [0, 0, 0, 0]
                # load remaining history for non-main agent
                for i in range(4):
                    if i != historical_winner_id:
                        player_id_action_counter = action_pointer[i]
                        agents[i].load_historical_actions(
                            action_history[i][player_id_action_counter:]
                        )
            else:
                winner_data_only = False
                # initialize environment as normal training sessions
                for i in range(4):
                    self.config["agent_clz{}".format(i)] = FeatureAgent2Adapted
                env = MahjongGBEnv(self.config)
                episode_data = {
                    agent_name: {
                        "state": {"observation": [], "action_mask": []},
                        "action": [],
                        "reward": [],
                        "value": [],
                        "info": [],
                    }
                    for agent_name in env.agent_names
                }
                obs = env.reset()
                agents = env.return_agent_list()
                done = False

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
                player_name_list = list(obs.keys())
                player_id_list = [int(a[-1]) - 1 for a in player_name_list]
                for i in range(len(player_id_list)):
                    player_id = player_id_list[i]
                    if not winner_data_only or player_id == historical_winner_id:
                        # infer with network
                        agent_name = player_name_list[i]
                        agent_data = episode_data[agent_name]
                        state = obs[agent_name]
                        agent_data["state"]["observation"].append(state["observation"])
                        agent_data["state"]["action_mask"].append(state["action_mask"])
                        state["observation"] = torch.tensor(
                            state["observation"], dtype=torch.float32
                        ).unsqueeze(0)
                        state["action_mask"] = torch.tensor(
                            state["action_mask"], dtype=torch.float32
                        ).unsqueeze(0)

                        with torch.no_grad():
                            logits, value = model(state)
                            action_dist = torch.distributions.Categorical(logits=logits)
                            if self.eval:
                                action = torch.argmax(logits, dim=1).item()
                            else:
                                action = action_dist.sample().item()
                            value = value.item()
                        # use correct action
                        if correct_action_flag:
                            # calculate correct action
                            player_id_action_counter = action_pointer[player_id]
                            action_pointer[player_id] += 1
                            if player_id_action_counter < len(
                                action_history[player_id]
                            ):
                                correct_action = action_history[player_id][
                                    player_id_action_counter
                                ]
                                action = correct_action
                        actions[agent_name] = action
                        values[agent_name] = value
                        agent_data["action"].append(actions[agent_name])
                        agent_data["value"].append(values[agent_name])
                        agent_data["info"].append(
                            [
                                player_id,
                                -1 if not correct_action_flag else historical_winner_id,
                                self.model_tag_id,
                                augmentation_key if not self.self_play else -1,
                            ]
                        )
                    else:
                        # follow historical actions if possible
                        action = agents[player_id].follow_history_or_rand()
                        actions[player_name_list[i]] = action

                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]["reward"].append(rewards[agent_name])
                obs = next_obs
                game_length += 1
            fan_list = env.get_fan_names()

            if winner_data_only:
                for i in range(4):
                    if i != historical_winner_id:
                        key_name = "player_{}".format(i + 1)
                        del episode_data[key_name]

            # gather necessary data
            # game_length, sampled_steps(total), win/loss
            if winner_data_only:
                win = list(rewards.values())[historical_winner_id] > 0
                winner_id = historical_winner_id
            else:
                win = max(list(rewards.values())) > 0
                if win:
                    winner_id = np.argmax(list(rewards.values()))
                else:
                    winner_id = -1

            # postprocessing episode data for each agent
            sampled_steps = 0
            for agent_name, agent_data in episode_data.items():
                if self.debug:
                    break
                if len(agent_data["action"]) < len(agent_data["reward"]):
                    agent_data["reward"].pop(0)
                if len(agent_data["state"]["observation"]) > 0:
                    obs = np.stack(agent_data["state"]["observation"])
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
                    if self.self_play:
                        info_list[:, 1] = winner_id
                    if not self.debug and not self.eval:
                        # send samples to replay_buffer (per agent)
                        self.replay_buffer_act.push(
                            {
                                "state": {"observation": obs, "action_mask": mask},
                                "action": actions,
                                "adv": advantages,
                                "target": td_target,
                                "info": info_list,  # player_id, winner_id, tag, augmentation scheme id
                            }
                        )
                    sampled_steps += len(actions)
            self.replay_buffer_act.push(
                {
                    "log": {
                        "win": win,
                        "eval": self.eval,
                        "guided": self.guide_trajectory,
                        "sampled_step": sampled_steps,
                        "game_length": game_length,
                        "reward": sum(agent_data["reward"]),
                        "fan_list": fan_list,
                    }
                }
            )

            # conpensate for faster iterations
            if self.guide_trajectory:
                time.sleep(random.uniform(0.15, 0.3))
        if not (self.debug):
            self.shm.close()
