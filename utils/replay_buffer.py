from multiprocessing import Manager
import _pickle as cPickle
from collections import deque, defaultdict
import numpy as np
import random
from bisect import bisect_right
import os
import json


def buffer_factory(capacity):
    return {"queue": deque(maxlen=capacity), "counter": 0}


class ReplayBufferSIL:
    """
    ReplayBufferSIL is specifically designed for MCR MahJong with 80 achievable fans
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # self.queue: temporary queue for holding data
        self.queue = Manager().Queue(capacity * 10)
        # actual struct for organizing gameplay records
        self.fan_dict = defaultdict(self._get_buffer)
        # meta data for counting records w.r.t. fans
        self.meta_counter = []
        self.meta_prefix_sum = []
        self.meta_keys = []

    def _get_buffer(self):
        return buffer_factory(self.capacity)

    def push(self, game_record):  # only called by actors
        self.queue.put(game_record)

    def get_entry_count(self):
        if len(self.meta_prefix_sum) == 0:
            return 0
        else:
            return self.meta_prefix_sum[-1]

    def organize(self):  # cache data from temporary queue and organize according to fan
        while not self.queue.empty():
            gameplay_record = self.queue.get()
            fan_list = gameplay_record["fans"]
            for fan in fan_list:
                self.fan_dict[fan]["queue"].append(gameplay_record)
                if self.fan_dict[fan]["counter"] < self.capacity:
                    self.fan_dict[fan]["counter"] += 1
        # update meta data
        self.meta_keys = list(self.fan_dict.keys())
        self.meta_counter = []
        for fan in self.meta_keys:
            self.meta_counter.append(self.fan_dict[fan]["counter"])
        # prefix sum
        self.meta_prefix_sum = np.concatenate(([0], np.cumsum(self.meta_counter)))

    def sample(self):  # sample trajectory according to meta data
        # randomly select trajectory
        sample_count = self.meta_prefix_sum[-1]
        sample_index = random.randint(0, sample_count - 1)
        # locate index
        fan_index = bisect_right(self.meta_prefix_sum, sample_index) - 1
        record_index = sample_index - self.meta_prefix_sum[fan_index]
        fan_name = self.meta_keys[fan_index]
        # access data
        trajectory_data = self.fan_dict[fan_name]["queue"][record_index]
        return trajectory_data

    def save_cache(self, save_path, checkpoint_name):
        # Save the current state of the replay buffer to a file
        with open(os.path.join(save_path, checkpoint_name), "wb") as f:
            cPickle.dump(self.fan_dict, f)

    def load_cache(self, save_path, checkpoint_name):
        with open(os.path.join(save_path, checkpoint_name), "rb") as f:
            self.fan_dict = cPickle.load(f)


class ReplayBufferActorSide:
    def __init__(self, capacity):
        self.queue = Manager().Queue(capacity)
        self.capacity = capacity
        self.buffer = []

    def empty(self):
        return self.buffer == []

    # Actors push data into queue
    def push(self, data):
        self.queue.put(data)

    # Manager get data from queue
    def get(self):
        while not self.queue.empty():
            data = self.queue.get()
            self.buffer.append(data)

    # Send data to learner
    def send_data(self):
        send_queue = self.buffer
        self.buffer = []
        return send_queue

    def send_data_pack(self, length=4):
        """
        Split data queue into chunks to optimize transmission
        Higher-length chunks have higher risk of lossing data
        """
        ret_data_pack = []
        send_queue = self.buffer
        self.buffer = []
        for i in range(0, len(send_queue), length):
            x = i
            ret_data_pack.append(send_queue[x : x + length])
        return ret_data_pack


class ReplayBufferLearnerSide:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rcvr_buffer = []
        self.buffer = deque(maxlen=self.capacity)
        self.stats = {"sample_in": 0, "sample_out": 0, "episode_in": 0}

    def rcv_data(self, data_list):
        """
        data_list: unprocessed data from networking components
        """
        rcvd_sample_count = 0
        for data in data_list:
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            rcvd_sample_count += len(unpacked_data)
            self.buffer.extend(unpacked_data)
            self.stats["sample_in"] += len(unpacked_data)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def rcv_data_filtered(self, data_list):
        """
        data_list: unprocessed data from networking components
        specifically filter out one-action-only actions
        """
        rcvd_sample_count = 0
        for data in data_list:
            unpacked_data_filtered = []
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            for datum in unpacked_data:
                if sum(datum["state"]["action_mask"]) > 1:
                    unpacked_data_filtered.append(datum)
            rcvd_sample_count += len(unpacked_data_filtered)
            self.buffer.extend(unpacked_data_filtered)
            self.stats["sample_in"] += len(unpacked_data_filtered)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def sample(self, batch_size):  # only called by learner
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats["sample_out"] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch

    def size(self):  # Need to be called after rcv_data() call
        return len(self.buffer)

    def clear(self):  # Need to be called after rcv_data() call
        self.buffer.clear()

    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res:
                    res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)

    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)


class ReplayBufferLearnerSideTagged:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rcvr_buffer = []
        self.buffer = deque(maxlen=self.capacity)
        self.stats = {"sample_in": 0, "sample_out": 0, "episode_in": 0}
        self.tag_count = defaultdict(int)
        self.max_tag_id = 0

    def rcv_data(self, data_list):
        """
        data_list: unprocessed data from networking components
        """
        rcvd_sample_count = 0
        for data in data_list:
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            rcvd_sample_count += len(unpacked_data)
            self.buffer.extend(unpacked_data)
            self.stats["sample_in"] += len(unpacked_data)
            self.stats["episode_in"] += 1
        return rcvd_sample_count

    def rcv_data_filtered(self, data_list):
        """
        data_list: unprocessed data from networking components
        specifically filter out one-action-only actions
        """
        rcvd_sample_count = 0
        for data in data_list:
            unpacked_data_filtered = []
            episode_data = data
            unpacked_data = self._unpack(episode_data)
            for datum in unpacked_data:
                if sum(datum["state"]["action_mask"]) > 1:
                    unpacked_data_filtered.append(datum)
                    tag_id = datum["info"][2]
                    self.tag_count[tag_id] += 1
                    if tag_id > self.max_tag_id:
                        self.max_tag_id = tag_id
            rcvd_sample_count += len(unpacked_data_filtered)
            self.buffer.extend(unpacked_data_filtered)
            self.stats["sample_in"] += len(unpacked_data_filtered)
            self.stats["episode_in"] += 1
        self.tag_count[self.max_tag_id - 2] = 0
        del self.tag_count[self.max_tag_id - 2]
        return rcvd_sample_count, self.tag_count

    def sample(self, batch_size):  # only called by learner
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats["sample_out"] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch

    def size(self):  # Need to be called after rcv_data() call
        return len(self.buffer)

    def clear(self):  # Need to be called after rcv_data() call
        self.buffer.clear()

    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res:
                    res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)

    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)


if __name__ == "__main__":
    buffer_sil = ReplayBufferSIL(100)
    buffer_sil.fan_dict["五门齐"]["queue"].append(1)
    buffer_sil.save_cache("./", "tmp.pkl")
    new_bf = ReplayBufferSIL(20)
    new_bf.load_cache("./", "tmp.pkl")
    print(new_bf.fan_dict)
